"""
Servidor completo para llamadas salientes (OUTBOUND) usando Twilio Media Streams
+ FastAPI WebServer
+ Endpoint /call para iniciar llamadas salientes vía Twilio REST API
+ Endpoint /outbound-call (TwiML) que Twilio solicita cuando inicia la llamada
+ WebSocket /media-stream que recibe G.711 μ-law desde Twilio, transcribe con Whisper
  (local), envía texto a Groq (Llama-3) y sintetiza respuesta con Coqui TTS (local)

Instrucciones rápidas:
1) Crear archivo .env con las variables listadas más abajo.
2) Instalar dependencias (requirements.txt sugerido).
3) Ejecutar: python twilio_groq_outbound_es.py
4) Asegúrate que el puerto y dominio sean accesibles desde Twilio (HTTPS obligatorio).

NOTAS:
- Este script asume ejecución en un entorno Linux/Unix donde se pueden instalar
  paquetes necesarios (whisper, TTS de Coqui, groq client si existe etc.).
- Ajusta nombres de modelo TTS y modelo Groq según tu cuenta/plan.
- Twilio seguirá cobrando por minutos de llamada. Groq/Whisper/Coqui pueden tener
  opciones gratuitas o locales; revisa sus licencias.

Variables de entorno requeridas (.env):
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_PHONE_NUMBER
GROQ_API_KEY
HOSTNAME  # dominio público (ej: example.com) o IP; usado para TwiML y WebSocket url
PORT (opcional, default 8000)

"""

import os
import json
import base64
import asyncio
import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from twilio.rest import Client
import uvicorn

# codecs/audio
import audioop

# Modelos/AI (puedes cambiar a llamadas a API en vez de locales)
# whisper (local) para transcripción
# pip install -U openai-whisper (o whisperx) o usa whisper de openai alternativa
try:
    import whisper
except Exception as e:
    whisper = None

# Coqui TTS
try:
    from TTS.api import TTS
except Exception as e:
    TTS = None

# Groq client placeholder; remplace con la librería oficial si difiere
try:
    from groq import Groq
except Exception as e:
    Groq = None

# ---------------------------
# Config y logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("twilio_groq")

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HOSTNAME = os.getenv("HOSTNAME", "localhost")
PORT = int(os.getenv("PORT", 8000))

if not TWILIO_SID or not TWILIO_AUTH or not TWILIO_NUMBER:
    log.warning("Faltan credenciales de Twilio en las variables de entorno.")

if not GROQ_API_KEY:
    log.warning("No se encontró GROQ_API_KEY; asegúrate de setearla si usarás Groq.")

# Inicializa clientes si están disponibles
twilio_client = None
if TWILIO_SID and TWILIO_AUTH:
    twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

groq_client = None
if Groq and GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Cargar Whisper local (opcional, pero recomendado para evitar costes)
whisper_model = None
if whisper is not None:
    try:
        log.info("Cargando modelo Whisper (small). Esto puede tardar...)")
        whisper_model = whisper.load_model("small")
    except Exception as e:
        log.exception("No se pudo cargar Whisper local: %s", e)

# Cargar TTS local (Coqui)
tts = None
if TTS is not None:
    try:
        # Cambia por un modelo multilingüe/ES real si lo prefieres
        tts = TTS("tts_models/multilingual/multi-dataset/your_tts")
    except Exception as e:
        log.exception("No se pudo cargar Coqui TTS: %s", e)

app = FastAPI()

# ---------------------------
# Helpers para μ-law <-> PCM
# ---------------------------

def decode_mulaw(ulaw_bytes: bytes) -> bytes:
    """Convertir bytes G.711 μ-law a PCM16 (little endian)"""
    try:
        pcm = audioop.ulaw2lin(ulaw_bytes, 2)  # 2 = width bytes per sample
        return pcm
    except Exception as e:
        log.exception("Error en decode_mulaw: %s", e)
        raise


def encode_mulaw(pcm_bytes: bytes) -> bytes:
    """Convertir PCM16 a G.711 μ-law"""
    try:
        ulaw = audioop.lin2ulaw(pcm_bytes, 2)
        return ulaw
    except Exception as e:
        log.exception("Error en encode_mulaw: %s", e)
        raise

# ---------------------------
# Endpoints para outbound
# ---------------------------

class CallRequest(BaseModel):
    to: str  # número destino en formato E.164: +521XXXXXXXXXX


@app.post("/call")
async def make_call(payload: CallRequest):
    """Inicia una llamada saliente usando Twilio REST API.
    Twilio pedirá luego la URL definida en `url` para obtener TwiML y crear el Media Stream.
    """
    if not twilio_client:
        return {"error": "Twilio client no configurado (ver variables de entorno)"}

    to = payload.to
    url = f"https://{HOSTNAME}:{PORT}/outbound-call"

    log.info("Creando llamada a %s usando TwiML: %s", to, url)

    call = twilio_client.calls.create(
        to=to,
        from_=TWILIO_NUMBER,
        url=url
    )

    return {"status": "calling", "sid": call.sid}


@app.get("/outbound-call")
@app.post("/outbound-call")
async def outbound_call(request: Request):
    """TwiML que Twilio solicita cuando crea la llamada. Contiene un <Connect><Stream/></>"""
    host = request.url.hostname or HOSTNAME
    # Mensaje inicial en español
    twiml = f"""
    <Response>
        <Say voice="alice" language="es-MX">Hola, te estamos llamando para conectarte con nuestro asistente. Esta llamada puede ser grabada.</Say>
        <Connect>
            <Stream url=\"wss://{host}/media-stream\" />
        </Connect>
    </Response>
    """
    return HTMLResponse(content=twiml, media_type="application/xml")


@app.get("/")
async def index():
    return "Servidor Twilio + Groq listo"

# ---------------------------
# WebSocket /media-stream
# ---------------------------

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Recibe mensajes del Media Stream de Twilio.
    Eventos que Twilio envía: start, media, stop, etc.

    - media: contiene payload base64 con G.711 μ-law
    """
    await websocket.accept()
    log.info("WebSocket aceptado desde Twilio")

    stream_sid = None
    audio_chunks = []  # lista de bytes μ-law

    async def receive_from_twilio():
        nonlocal stream_sid, audio_chunks
        try:
            async for raw_msg in websocket.iter_text():
                data = json.loads(raw_msg)
                evt = data.get("event")

                if evt == "start":
                    stream_sid = data.get("start", {}).get("streamSid")
                    log.info("Stream started %s", stream_sid)

                elif evt == "media":
                    payload_b64 = data.get("media", {}).get("payload")
                    if not payload_b64:
                        continue
                    try:
                        ulaw = base64.b64decode(payload_b64)
                        # append raw μ-law bytes
                        audio_chunks.append(ulaw)
                    except Exception as e:
                        log.exception("Error decoding base64 media: %s", e)

                elif evt == "stop":
                    log.info("Stream stopped by Twilio %s", stream_sid)
                    # break read loop? continue to allow graceful close

        except WebSocketDisconnect:
            log.info("Twilio WebSocket desconectado")

    async def process_loop():
        nonlocal audio_chunks, stream_sid

        while True:
            await asyncio.sleep(1.0)

            if not audio_chunks:
                continue

            # Unir y vaciar buffer μ-law
            ulaw_raw = b"".join(audio_chunks)
            audio_chunks = []

            try:
                pcm = decode_mulaw(ulaw_raw)
            except Exception as e:
                log.exception("No se pudo decodificar μ-law: %s", e)
                continue

            # ---- Transcribir (Whisper local si está disponible) ----
            user_text = ""
            if whisper_model is not None:
                try:
                    # Whisper acepta archivos o numpy arrays; como simplificación, guardamos temporal
                    tmp_wav = "tmp_input.wav"
                    # Guardar pcm como wav
                    import wave
                    with wave.open(tmp_wav, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(8000)  # Twilio G.711 μ-law es mono 8kHz
                        wf.writeframes(pcm)

                    res = whisper_model.transcribe(tmp_wav, fp16=False, language="es")
                    user_text = res.get("text", "").strip()
                except Exception as e:
                    log.exception("Error en transcripción Whisper: %s", e)
            else:
                log.warning("Whisper no disponible — no se transcribe audio")

            if not user_text:
                log.info("Transcripción vacía — se omite ciclo")
                continue

            log.info("Usuario dijo: %s", user_text)

            # ---- LLM: enviar a Groq ----
            ai_text = ""
            if groq_client is not None:
                try:
                    completion = groq_client.chat.completions.create(
                        model="llama3-70b-8192",  # ajusta según tu cuenta
                        messages=[
                            {"role": "system", "content": (
                                "Eres un asistente de voz en tiempo real. Responde en espa\u00f1ol. "
                                "Respuestas breves y amistosas. Mantén respuestas <5s.")},
                            {"role": "user", "content": user_text}
                        ]
                    )
                    ai_text = completion.choices[0].message.get("content", "").strip()
                except Exception as e:
                    log.exception("Error llamando a Groq: %s", e)

            else:
                # Fallback simple si no hay Groq: eco corto
                ai_text = f"Entiendo: {user_text}. ¿En qué más puedo ayudar?"

            log.info("AI responde: %s", ai_text)

            # ---- TTS: generar WAV ----
            wav_bytes = None
            try:
                if tts is not None:
                    tmp_out = "response.wav"
                    # Intenta elegir voz en español; si no existe, usa default
                    try:
                        tts.tts_to_file(text=ai_text, file_path=tmp_out, speaker="female-es")
                    except Exception:
                        # intento sin especificar speaker
                        tts.tts_to_file(text=ai_text, file_path=tmp_out)

                    with open(tmp_out, "rb") as f:
                        wav_bytes = f.read()
                else:
                    # Fallback: generar un WAV muy simple (silencio + TTS no disponible)
                    import wave
                    tmp_out = "response.wav"
                    with wave.open(tmp_out, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(b"\x00\x00" * 16000)
                    with open(tmp_out, "rb") as f:
                        wav_bytes = f.read()
            except Exception as e:
                log.exception("Error en TTS: %s", e)

            if not wav_bytes:
                log.warning("No se generó audio de TTS")
                continue

            # ---- Convertir WAV (PCM16 16kHz probablemente) a μ-law 8kHz para Twilio ----
            try:
                # Leemos WAV y convertimos sample rate si es necesario
                import io
                import wave
                wav_buf = io.BytesIO(wav_bytes)
                with wave.open(wav_buf, "rb") as wf:
                    nch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())

                # Si sr != 8000 convertimos con audioop.ratecv
                if sr != 8000:
                    # convertir a mono 16-bit si necesario
                    if nch > 1:
                        frames = audioop.tomono(frames, sw, 1, 0)
                    # re-muestrear
                    frames, _ = audioop.ratecv(frames, sw, 1, sr, 8000, None)
                    sr = 8000

                # ahora frames es PCM16 8kHz mono: convertir a μ-law
                ulaw_resp = encode_mulaw(frames)

                b64 = base64.b64encode(ulaw_resp).decode("utf-8")

                # Enviar evento media a Twilio via websocket
                await websocket.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": b64}
                })

            except Exception as e:
                log.exception("Error al convertir WAV -> μ-law y enviar: %s", e)

    # correr tasks
    try:
        await asyncio.gather(receive_from_twilio(), process_loop())
    except Exception as e:
        log.exception("Error en media_stream: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------
# Lanzador
# ---------------------------
if __name__ == "__main__":
    # Nota: Twilio requiere HTTPS; para pruebas locales usa ngrok para exponer tu servidor con HTTPS
    log.info("Iniciando servidor en 0.0.0.0:%s", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
