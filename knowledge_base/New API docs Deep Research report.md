# **Hume AI Platform – Comprehensive API Documentation**

Hume AI is an emotion-centric AI platform providing advanced models to analyze and generate expressive human communication . Hume’s APIs span **Expression Measurement**, the **Empathic Voice Interface (EVI)**, and **Octave Text-to-Speech (TTS)**, enabling applications to measure emotions in voice, facial expressions, and language, or synthesize speech with rich emotional nuance. This documentation covers every Hume API, detailing features, endpoints, parameters, WebSocket channels, authentication, integration options, and usage examples in Python. By the end, you should be able to confidently integrate Hume AI into real-time and batch workflows without further research.

## **Expression Measurement API (Multimodal Emotion Analysis)**

Hume’s **Expression Measurement API** analyzes emotional expression from four input modalities – **text, audio, video, and images** – yielding hundreds of dimensional scores that quantify nuanced emotions and expressive behaviors . It encapsulates state-of-the-art models for:

* **Facial Expression** – Detects subtle facial movements (smiles, frowns, brow raises, etc.) and their emotional meaning along 48 dimensions . Optionally, it can output **FACS 2.0** (Facial Action Coding System) metrics (e.g. Action Units like “Inner Brow Raise,” “Nose Crinkle,” plus gestures like “Hand over mouth”) .

* **Speech Prosody** – Captures the non-linguistic tone, rhythm, and timbre of speech (how something is said) along 48 dimensions , revealing emotional cues from voice modulation.

* **Vocal Bursts** – Detects non-verbal vocal expressions (laughter, sighs, cries, “uh-huh,” etc.) along 48 dimensions , indicating emotional states in short vocal sounds.

* **Emotional Language** – Analyzes the emotional tone of text or transcribed speech along 53 dimensions , reflecting sentiment and nuanced affective content of language.

**Emotion Taxonomy:** The models output continuous **confidence scores (0 to 1\)** for each emotion dimension, indicating the intensity/likelihood that a human would perceive that emotion in the input . Higher values mean the expression strongly exhibits traits associated with that emotion label . Emotions are labeled with everyday terms (e.g. *joy, anger, confusion, empathy, pride*, etc.) for interpretability, although they represent complex blends of features . The language model produces 53 possible emotion labels, and the face/prosody/burst models share a set of 48 labels (largely overlapping) . For reference, some of the expressions measured include *Admiration, Amusement, Anger, Calmness, Confusion, Disgust, Fear, Joy, Sadness, Surprise (positive/negative),* and dozens more . Each input may score highly on multiple dimensions simultaneously. The scores represent **perceptual intensity** – e.g. a value of 0.8 on “Amusement” means an average human rater would strongly perceive amusement in the input . Because expression is context-dependent and multimodal, **face and voice outputs may differ** (e.g. a person’s tone may convey *enthusiasm* while their face shows *awkwardness*); such differences are expected and underscore that these scores are not a direct readout of internal feelings .

**Granularity Options:** For spoken language and prosody analysis, the API supports multiple time granularities to suit your application . You can configure whether speech is analyzed **per word**, **per sentence**, **per utterance**, or **per entire conversational turn**:

* *Word-level:* Each word is assigned its own set of emotion scores, providing very fine-grained insight .

* *Utterance-level:* \~1–2 second segments (natural pauses) are analyzed, updating emotion measures more frequently than full sentences . (For text inputs, utterance-level is equivalent to sentence-level output .)

* *Sentence-level:* Each sentence yields an aggregated emotion score set , useful for aligning emotions to phrases or lines of dialogue.

* *Conversational turn:* Each speaker turn (from when one person starts speaking until they stop) produces one set of scores . This higher-level view is useful in multi-speaker conversations to get the overall tone of each turn.

These options let you balance **temporal resolution** versus **stability** of the measurements . By default, the batch API returns sentence-level or whole-text results for language, and segment-based results for audio/video, but you can specify the desired granularity in the request configuration.

### **Batch Expression Measurement API (Asynchronous REST)**

Hume’s batch API is designed for **large-scale, offline processing** of media files and text. It uses a **job-based asynchronous workflow** to handle multiple inputs in parallel and return results when analysis completes . This is ideal for processing datasets or archives of recordings (e.g. a collection of call center audio or user videos) without blocking your application.

**Endpoint:** POST https://api.hume.ai/v0/batch/jobs – create a new analysis job. The request must include your API key and specify which models to run. For example, to analyze facial expressions in a set of images, you could send:

curl https://api.hume.ai/v0/batch/jobs \\  
  \--request POST \\  
  \--header "Content-Type: application/json" \\  
  \--header "X-Hume-Api-Key: \<YOUR\_API\_KEY\>" \\  
  \--data '{  
    "models": {  
        "face": { "facial\_action\_units": false }  
    },  
    "urls": \[  
        "https://example.com/images/meeting1.jpg",  
        "https://example.com/images/meeting2.jpg"  
    \]  
  }'

In this JSON payload, the "models" object selects which analysis models to apply. Here we choose the face model (facial expression analysis). We could similarly include "prosody": {}, "burst": {}, and "language": {} under "models" to run those in the same job. Each model may accept configuration options – for instance, the face model could set "facial\_action\_units": true to include FACS action unit outputs, and the language model could set a "granularity" or a specific "language" code for transcription. If analyzing audio where you want text analysis too, you can include both "prosody": {} and "language": {}; Hume will automatically transcribe the speech (with language auto-detection or a provided locale) to feed the language model . You can also configure a "transcription" object with a fixed language (BCP-47 code) to improve ASR accuracy . The above example uses a "urls" list to point to input files; you can provide data in three ways:

* **Public URLs:** Include a "urls" array of file links (HTTP(S) accessible). Each URL can be an **image**, **audio**, **video**, or **text** file (or even a **.zip/.tar** archive of multiple files). This is convenient for cloud-hosted data. *(Max URL file size: 1 GB) .*

* **Direct File Upload:** Post multipart form-data with one or more "file" fields. For example, using cURL: \--form file=@audio1.wav \--form file=@video1.mp4 \--form json='{"models": {"prosody":{} }}'. The "json" field holds the same JSON config as above, while the file fields carry binary data . Use this for local files. *(Max uploaded file size: 100 MB each) .*

* **Raw Text:** Provide a "texts" (or "strings") array in the JSON with plain text strings to analyze . Each entry is treated as a separate document for the language model. *(Max length per text string: 255 MB of text) .*

You can mix input types in one job (e.g. a few text strings, some image URLs, and a couple of uploaded audio files in one request) – up to **100 items total** per job across all input lists . This flexibility allows aggregating varied data for one batch analysis.

**Job Processing:** When you create a job, the API responds immediately with a job\_id and initial status. The job then runs asynchronously on Hume’s servers. You should **avoid polling** too frequently for results; instead, Hume offers a **webhook callback** mechanism to notify you on completion :

* To use webhooks, include a "callback\_url" in your POST payload (or you can PATCH it later). For example: "callback\_url": "https://your.server.com/hume\_callback". When the job finishes, Hume will send an HTTP POST to your URL containing a JSON body with the job\_id, status ("completed" or "failed"), and the predictions if completed .

* Alternatively, you can manually retrieve status and results. Use GET https://api.hume.ai/v0/batch/jobs/\<JOB\_ID\> for status details , and GET https://api.hume.ai/v0/batch/jobs/\<JOB\_ID\>/predictions for the results JSON . There’s also an endpoint to download all results in a single CSV or NDJSON file, and another to fetch any output artifacts (e.g. if you requested waveform audio of laughter segments, etc.) .

**Response Schema:** When completed, each job yields a list of **predictions** corresponding to your inputs. The JSON result has the structure:

{  
  "job\_id": "...",  
  "status": "COMPLETED",  
  "predictions": \[  
    {  
      "input\_id": "\<ID or URL of input\>",  
      "models": {  
        "face": { ... facial expression results ... },  
        "prosody": { ... vocal prosody results ... },  
        "language": { ... text emotion results ... }  
      }  
    },  
    {  
      "...": "..."  
    }  
  \]  
}

Each input’s results are organized by model. For **image inputs**, the face model returns either one set of scores per image (or multiple if multiple faces detected – face outputs include a face identifier if more than one face is present). For **audio/video**, the prosody and burst models produce a time-series: typically an array of segments (with time timestamps or indices) each containing the 48-dimensional emotion scores for that segment. The language model on audio transcriptions can output sentence or word-level scores as configured. For example, an audio file might result in { "prosody": { "predictions": \[ { "start\_ms":0, "end\_ms":1000, "emotions": {...48 scores...} }, { "start\_ms":1000, ...} \] } } and likewise a "language" section if speech was transcribed. **Confidence scores** are typically in \[0,1\] (though values may rarely exceed 1.0 for very strong expressions due to the modeling approach) . If an input had no detectable signal for a model (e.g. no face in an image, or no vocal burst in audio), that model’s result may be empty and a warning like “No faces detected” or “No vocal bursts detected” will be included .

*As shown above, the Batch API follows a job workflow:* (1) You submit a job with input data and model config; (2) Hume processes the inputs, possibly in parallel; (3) you retrieve results via webhook or polling. This asynchronous pattern allows you to queue large jobs (up to 500 jobs can be in the queue at once) without blocking. Each job can handle significant media: up to 3 hours of audio/video per file . If you need to process more, you can chunk the media or use multiple jobs.

**Batch Example – Analyzing Sales Call Recordings:** Below is a **Python** example (using requests) that submits a batch job to analyze a set of customer service call audios for emotional content in voices, then fetches results:

import requests, time

API\_KEY \= "YOUR\_API\_KEY"  
headers \= {"X-Hume-Api-Key": API\_KEY}

\# 1\. Submit a new batch job with two audio files (URLs or local paths)  
job\_request \= {  
    "models": {  
        "prosody": {},      \# voice tone analysis  
        "language": {}      \# transcribed content sentiment  
    },  
    "urls": \[  
        "https://hume-example-data.s3.amazonaws.com/call1.wav",  
        "https://hume-example-data.s3.amazonaws.com/call2.wav"  
    \]  
}  
resp \= requests.post("https://api.hume.ai/v0/batch/jobs", json=job\_request, headers=headers)  
resp.raise\_for\_status()  
job\_id \= resp.json()\["job\_id"\]  
print(f"Submitted job {job\_id}")

\# 2\. Wait for completion (polling every 10 seconds as example; in production use webhook callbacks)  
status \= "PENDING"  
while status not in ("COMPLETED", "FAILED"):  
    time.sleep(10)  
    status\_resp \= requests.get(f"https://api.hume.ai/v0/batch/jobs/{job\_id}", headers=headers)  
    status \= status\_resp.json().get("status")  
    print(f"Job status: {status}")

\# 3\. Retrieve results if completed  
if status \== "COMPLETED":  
    result\_resp \= requests.get(f"https://api.hume.ai/v0/batch/jobs/{job\_id}/predictions", headers=headers)  
    results \= result\_resp.json()\["predictions"\]  
    for i, pred in enumerate(results):  
        call\_url \= job\_request\["urls"\]\[i\]  
        prosody\_scores \= pred\["models"\]\["prosody"\]\["predictions"\]  \# list of segments with emotion scores  
        language\_scores \= pred\["models"\]\["language"\]\["predictions"\]  \# sentiment per transcript segment  
        print(f"Results for call {call\_url}:")  
        \# Example: print average frustration vs calmness in voice  
        avg\_frustration \= sum(seg\["emotions"\]\["Frustration"\] for seg in prosody\_scores) / len(prosody\_scores)  
        avg\_calm \= sum(seg\["emotions"\]\["Calmness"\] for seg in prosody\_scores) / len(prosody\_scores)  
        print(f" Average Frustration={avg\_frustration:.2f}, Calmness={avg\_calm:.2f}")  
        \# ... further processing ...

In this code, we request two models on each audio: prosody (to gauge vocal tone, e.g. frustration) and language (to gauge sentiment from spoken words). The API will automatically transcribe the audio for the language model. We poll for simplicity, but **in production we’d set a callback URL** so Hume notifies our server when done . Once completed, we fetch the predictions. We then parse the results (for brevity we computed average frustration/calmness; real use might store the full time-series or detect moments of high distress for escalation).

**Batch API Limits & Error Handling:** The batch service has generous limits, but be mindful of them when designing your pipeline :

* **File size limits:** 1 GB per file for URL inputs; 100 MB per file for direct upload . Archives (ZIP, TAR) count as one file toward this limit. Large datasets should be chunked.

* **Content length limits:** Audio/video max \~3 hours each . Text inputs max \~255 MB, which is about 100 million characters – effectively no limit for typical use .

* **Job concurrency:** Up to 500 pending/running jobs at once per account . If you exceed, you’ll get an error and should wait for some jobs to finish before submitting more.

* **Output size:** The JSON results can be large if you process long media (emotion scores every few seconds). Consider using the CSV/NDJSON download if easier, or storing results in a database rather than holding in memory.

If the API encounters problems, it returns informative error codes. For example, if you submit an unsupported file format or a corrupt file, you might get E0200 (media not recognized) . If an audio file exceeds the length limit, the error will clearly state the 3-hour limit and the duration of your file . A few common error codes to note:

* E0102: Model incompatible with file (e.g. you requested face on a text input) .

* E0202: No audio in file (you asked for audio models but the video had no soundtrack) .

* E0401: File upload failed (did not meet format/size requirements) .

* W0101–W0105: Warnings (not errors) indicating nothing found – e.g. W0105 “No speech detected” if an audio was silent . These mean the job ran, but certain outputs are empty.

Always check the status field; on failure, the status will be "FAILED" and an error message will be provided. For partial warnings (e.g. one model fails), the job can still be "COMPLETED" but with warnings array present.

### **Real-Time Expression Measurement Streaming (WebSocket API)**

For **interactive or live use cases**, Hume provides a **WebSocket streaming API** to get immediate emotional analysis on streaming data . This is ideal for applications like live sentiment dashboards, real-time user feedback, or powering interactive experiences that react to a user’s emotions on the fly. The streaming API maintains an open socket for two-way communication, allowing you to continuously send data (text, audio, video frames) and receive model predictions in near real-time .

**Endpoint:** wss://api.hume.ai/v0/stream/models for a WebSocket handshake . Authentication is done by sending the API key in the headers during the handshake (or using a token, discussed later) . Once connected, the client will **send messages** containing data and config, and receive messages with predictions.

**Streaming Workflow:** After opening the WebSocket, you typically first send a **configuration message** indicating which models to run (similar to the "models" JSON used in batch). For example, you might send a JSON like: {"models": {"prosody": {}, "language": {}}, "transcription": {"language": "en"}} to analyze voice tone and language on an audio stream, forcing English transcription. The server will acknowledge, and then you can start streaming data. Key points:

* **Sending Audio/Video:** Audio data (e.g. from a microphone) should be sent in small chunks (to respect real-time constraints). The streaming API expects binary audio frames encoded as PCM WAV bytes (16-bit, typically 24kHz or 16kHz). Each audio message should be accompanied by the JSON payload specifying its type. In Hume’s Python SDK, for instance, you call socket.send\_file("audio\_chunk.wav") and it internally sends the bytes base64-encoded with the proper JSON structure . **Each audio message is limited to \~5 seconds of audio**; if you send a longer clip in one go, the socket will return an error E0203 (“audio file too long … limit is 5000 milliseconds”) . So, split live audio into 1–2 second chunks (or smaller) and send sequentially. The API will process each chunk and return a prediction for it. For continuous audio, you’ll get a stream of predictions, typically one per chunk.

* **Sending Video Frames:** For real-time video (facial expression analysis), it’s recommended to capture individual frames (images) and send them one by one rather than sending a whole video file . Each image should be JPEG/PNG bytes (the API supports up to 3000×3000 resolution; larger will error E0205) . You could send, say, \~1–5 FPS images to capture facial expressions. The API will return face analysis for each frame. (If the video also has audio and you want voice analysis too, you should send audio separately as above. Multi-modal synchronization is discussed shortly.)

* **Sending Text:** You can stream textual data as well. Simply send a JSON message with a field like "text": "Some input sentence." (the Hume Python SDK provides socket.send\_text("...") for this ). The language model will return the emotional analysis for each text chunk. This is useful if you have a live text stream (e.g. chat messages) to analyze in real-time.

**Receiving Predictions:** The server sends back a prediction message for each data message you sent, usually within a few hundred milliseconds. The structure mirrors the batch output but scoped to just that chunk. For example, sending one audio chunk might yield a message: {"language": { "predictions": \[ {...emotions...} \] }, "prosody": { "predictions": \[ {...} \] }} if you requested both models. If using the Python async SDK, result \= await socket.send\_text(...) will return a high-level object; e.g. result.language.predictions\[0\].emotions might be a dict of emotion scores . You handle these results as they arrive – e.g., update your UI or make decisions on the fly (like alert if “Distress” exceeds a threshold).

The streaming connection remains open for continuous analysis. This persistent two-way channel enables truly interactive applications: your client can decide to send more data based on prior results, etc., without the overhead of repeated HTTP requests .

**Streaming Example – Live Voice Emotion Display:** The following example demonstrates a simple Python client that streams audio from the microphone to Hume and prints out real-time emotion scores (note: for actual mic capture we’d use a library like pyaudio or sounddevice, but here we simulate by reading chunks from a WAV file for illustration):

import asyncio, websockets, base64, json

API\_KEY \= "YOUR\_API\_KEY"  
uri \= "wss://api.hume.ai/v0/stream/models"

async def stream\_audio():  
    async with websockets.connect(uri, extra\_headers={"X-Hume-Api-Key": API\_KEY}) as ws:  
        \# 1\. Send model configuration (request prosody model for vocal tone)  
        config\_msg \= {"models": {"prosody": {}}}  
        await ws.send(json.dumps(config\_msg))  
        \# 2\. Simulate streaming by reading an audio file in chunks  
        with open("caller\_audio.wav", "rb") as f:  
            CHUNK\_SIZE \= 16000  \# e.g. 1 second of 16kHz 16-bit audio \~ 32KB  
            chunk \= f.read(CHUNK\_SIZE)  
            while chunk:  
                \# Send audio chunk as base64 JSON message  
                audio\_b64 \= base64.b64encode(chunk).decode('ascii')  
                audio\_msg \= {"type": "audio", "data": audio\_b64}  
                await ws.send(json.dumps(audio\_msg))  
                \# Receive and parse prediction for this chunk  
                response \= await ws.recv()  
                result \= json.loads(response)  
                if "prosody" in result:  
                    emotions \= result\["prosody"\]\["predictions"\]\[0\]\["emotions"\]  
                    print(f"Voice emotions: {emotions}")  \# e.g. {'Calmness': 0.55, 'Anger': 0.12, ...}  
                elif "error" in result:  
                    print("Error from Hume:", result\["error"\])  
                    break  
                chunk \= f.read(CHUNK\_SIZE)  
        \# 3\. Optionally, close the session (Hume will also close if no activity).  
        await ws.close()

asyncio.run(stream\_audio())

In this snippet, we connect to the streaming endpoint with our API key. We immediately send a config message requesting only the prosody model. Then we read an audio file in small chunks and send each chunk base64-encoded as a JSON message with "type": "audio". After each send, we await a response. The response is parsed: if it contains a "prosody" key, we extract the emotions scores and print them. (In a real app, you might update a gauge or trigger some logic if e.g. “Excitement” rises.) If an "error" comes back, we break out. This loop effectively mimics live audio streaming. You would replace the file read with a microphone loop, and perhaps throttle to real-time. The result is a continuous printout of emotional probabilities for the caller’s voice, updated every second.

**WebSocket Error Handling & Reconnection:** In live systems, you must handle network drops or timeouts gracefully. Hume’s WebSocket will send a normal closure (1000 close\_normal) if the connection is inactive too long or exceeds the maximum session duration (currently the service may time out after some period of hours) . You should detect closure and can **reconnect** automatically if needed. Crucially, Hume supports **chat context resumption** for EVI (detailed in EVI section), but for pure expression streaming, context is usually not needed to resume – you can just open a new socket and continue streaming data. If a config message was not acknowledged or an error like 1008 policy\_violation (bad request) is received , correct your payload and reconnect. In our loop above, if ws.recv() raises due to disconnect, we’d catch it, log it, and perhaps call websockets.connect again after a short delay. Always ensure to resend the config message on a fresh connection before streaming data.

**Advanced: Multi-Modal Fusion & Synchronization:** If you want to analyze **multiple modalities simultaneously** (e.g. a video call with both audio and video), you have a couple of options:

* **Single Job/Stream, Multiple Models:** The batch API can automatically handle multi-modal inputs. For example, if you submit a video file and request both face and prosody models, Hume will internally extract frames for face analysis and audio for prosody. The outputs for both modalities will share a timeline (timestamps) so you can correlate them (e.g. see if a spike in voice anger coincided with a frown). In the streaming API, however, there is not a single “multiplexed” stream – you need to send images and audio as separate messages. You can interleave them on one WebSocket connection: e.g. send an image frame, then an audio chunk, etc. The responses will come back in the same order. It’s then up to you to align them by timestamps. For synchronization, timestamp each frame and audio chunk when you capture them, and when results return, attach those timestamps. Since each result includes timing (e.g. start\_ms for audio segments, and you can tag image messages with a custom ID), you can sync the streams in your application.

* **External Fusion Logic:** The emotion dimensions from face vs voice might both be relevant. For instance, if either the face *or* voice shows high “Distress”, you might treat the user as distressed overall. A simple fusion strategy is to take the maximum or average of corresponding emotion scores from face and voice at a given time window. More advanced techniques include training a model on the combined features – Hume’s **Custom Model API** allows you to feed in the raw expression features (from face, voice, text) and train a classifier or regressor for a particular outcome (e.g. train a model to predict customer satisfaction score from both facial and vocal cues). Custom Models essentially learn an optimal fusion of modalities for your target metric. For real-time fusion without training, you could also implement heuristic rules (e.g. if face OR voice indicates anger \>0.7, flag anger). Keep in mind that modalities can conflict, so decide which signals you trust more for each emotion (human studies show some emotions are more readily perceived via voice, others via face).

Finally, ensure to follow ethical guidelines when using these outputs . Hume provides extensive **scientific best practices** documentation – remember that expression ≠ inner feeling, and use aggregated insights responsibly.

## **Empathic Voice Interface (EVI – Speech-to-Speech Conversational AI)**

Hume’s **Empathic Voice Interface (EVI)** is a real-time, interactive AI agent that engages in spoken conversations with emotional intelligence . EVI is a **speech-to-speech** system: it listens to a user’s voice, understands not just the words but the prosody/emotion behind them, and responds with generated speech that is contextually and emotionally appropriate . Under the hood, EVI combines Hume’s expression measurement (for vocal prosody analysis) with a powerful large language model (LLM) and the Octave TTS model to generate its responses . The result is an AI that *sounds* like it understands you – modulating its tone, timing, and wording to convey empathy, turn-taking, and personality.

### **EVI Key Features and Capabilities**

**Core Dialogue Features:**

* **Automatic Speech Recognition (ASR) & Transcription:** EVI transcribes user speech in real-time with high accuracy, providing not only the text but also aligning Hume’s expression measures to each sentence . This means as the user speaks, EVI is extracting emotions from their tone concurrently with understanding the content.

* **Language Understanding & Generation (LLM):** EVI uses a specialized speech-optimized language model to decide *what* to say. It can integrate with various partner LLMs (Anthropic Claude, OpenAI GPT, Google PaLM, etc., or Hume’s own model) as configured . The LLM takes into account the user’s words *and* measured emotional state to produce an empathic textual response (for example, if it senses frustration, it might choose more apologetic phrasing) .

* **Expressive Voice Response (TTS):** EVI speaks the response aloud using Octave TTS, with a voice that can dynamically adapt in tone. The speech-language model guides prosody to match the user’s vibe – e.g. responding to user excitement with an upbeat tone, or speaking softly if the user sounds sad . The voice can be selected or custom-designed (more on voices below).

* **Low Latency, Turn-Taking & Interruptibility:** EVI is optimized for fast response. It performs end-of-turn detection on the user’s voice using prosody cues, so it knows when you’ve finished speaking (even without a hard stop) . It then begins responding immediately (minimizing lag) . Importantly, EVI is **always interruptible** – if the user starts talking over it, EVI will detect the interruption and stop speaking instantly to listen . This makes conversations feel natural, like speaking with a polite human who doesn’t talk over you.

**Empathic Abilities:** EVI’s distinguishing factor is emotional attunement :

* **Understands Vocal Emotion:** EVI continuously analyzes the user’s prosody (tone, pitch, intensity) using Hume’s models while they speak . These signals (e.g. stress, excitement, hesitation) are fed into the LLM’s context. This allows the AI to interpret *how* something is said. For example, if the user says “I’m fine” in a flat, sad tone, the system can infer they might not actually be fine and respond with more concern.

* **Adaptive Response Tone:** When EVI speaks, it doesn’t use a fixed robotic voice – it adapts its **vocal style** to suit the conversation. It will match the user’s energy level and emotional state to some degree (while following the configured personality). If a user sounds confused and hesitant, EVI might reply in a gentle, clarifying tone. It can express empathy explicitly in wording (*“I’m sorry to hear that.”*) and implicitly via vocal cues (speaking softly, slower, etc.) . EVI even adjusts **timing** – for instance, it’s been trained for excellent **end-of-turn prediction** so it doesn’t interrupt, and it inserts appropriate pauses or acknowledgments.

* **Context and Memory:** EVI can maintain context over a conversation. It tracks not only the content but the emotional context of the dialogue. It can resume a chat session with memory of prior interactions (see Chat persistence below), allowing multi-session continuity.

**Use Cases:** EVI unlocks use cases like: *interview simulators* (it responds to your answers with human-like empathy), *AI coaches/therapists* (listening for distress or progress), *virtual companions* for wellness or learning (that actively listen and respond in engaging ways), *customer support bots* that detect caller frustration and adjust tone, *in-car assistants* that sense driver stress, etc. .

### **EVI API Endpoints and Architecture**

EVI’s API primarily operates over a **WebSocket** for live audio streaming. It also provides REST endpoints for managing configurations and retrieving conversation logs. The key endpoints/components are:

* **WebSocket Chat Endpoint:** **wss://api.hume.ai/v0/evi/chat** – This is the main entry point for real-time EVI sessions. Clients connect here to send and receive audio/messages in a conversation . It’s a full-duplex channel for a single chat session.

* **Configurations (REST):** EVI allows you to create and manage **configurations** that define the bot’s behavior: including the base prompt (persona or instructions), the voice to use, the language model to use (and its API keys if external), and any integrated tools/webhooks. These can be managed via REST endpoints:

  * POST /v0/evi/configs to create a new config,

  * GET /v0/evi/configs/\<id\> to retrieve,

  * etc. (The docs refer to these under “Configs”). Each config gets a unique config\_id and a version number.

* **Chat History & Groups (REST):** EVI logs each chat session. You can list past chats (GET /v0/evi/chats) and retrieve transcripts or metadata. Chat sessions can be grouped by a persistent conversation ID (Chat Group) to allow continuity across sessions. Endpoints like GET /v0/evi/chat\_groups list all ongoing conversations (groups).

* **Webhooks & Tools:** EVI can be extended with custom **tools** (functions the LLM can call, such as querying a database) and can emit **webhook events** to your server for certain triggers (like conversation start, end, or custom events). These are configured in the EVI config. For example, you might set a webhook URL to get a JSON payload every time a chat starts/ends, for logging or analytics . You subscribe to events (like chat\_started, chat\_ended, or even per-message events) in the config, and EVI will POST to your endpoint in real-time as those occur, allowing server-side intervention or monitoring .

**Connecting to EVI (WebSocket):** To start a conversation, connect to wss://api.hume.ai/v0/evi/chat. You must authenticate – either by providing the API key or a token. **Authentication options:**

* *API Key (simple):* Include ?api\_key=\<YOUR\_API\_KEY\> as a query parameter, or send it in headers as X-Hume-Api-Key during the websocket handshake . This is fine for trusted environments (server side).

* *Access Token (secure for client use):* If you don’t want to expose the API key (e.g. in a web app), first obtain a temp **access token** via OAuth2 (server-to-server) and then connect with ?access\_token=\<TOKEN\> . (See **Auth** below for details on generating tokens.) If an access\_token is provided, you omit the api\_key.

You can also specify **which configuration** EVI should use: include config\_id=\<ID\> (and optionally config\_version=\<N\>) as query params in the URL . If not provided, EVI uses a default generic config. Using a custom config is how you give EVI a persona or custom behavior (for example, an upbeat customer service agent vs. a calm therapist require different prompts and voices). You can retrieve the config\_id from the Hume platform or via API after creating a config.

**WebSocket Communication – Messages:** Once connected, EVI and the client exchange messages in JSON. EVI’s WebSocket protocol is event-based with a variety of message types. Major message types include:

* **AudioInput (client → EVI):** Contains chunks of user’s audio. Format: {"type": "audio\_input", "data": "\<base64 audio chunk\>"} . You should stream the microphone audio in near-real-time in this format. EVI will process audio continuously; there’s no fixed limit stated for EVI audio chunks, but smaller chunks (\~100–500ms) help responsiveness. EVI begins processing even before you finish sending if possible, and will start formulating a response once it detects end-of-turn.

* **SessionSettings (client → EVI):** Used to send initial settings or adjustments mid-chat. This can include specifying a new voice, switching the language model, or other runtime options. Typically sent at start (or not at all if you rely on the config defaults).

* **UserInput (client → EVI):** If you prefer to send *text* instead of audio (say, for testing or if the user typed), you can send a message of type user\_input with a text field. EVI will treat it as if that was transcribed from speech.

* **AssistantMessage (EVI → client):** This carries the **assistant’s generated text** reply . It includes fields like text (the content of the response) and maybe metadata (e.g. which tool was used to generate it, if any). The assistant’s text is generated quickly thanks to the integrated LLM.

* **AssistantProsody (EVI → client):** Before or alongside sending audio, EVI provides an **AssistantProsody** message . This includes information about how the assistant will speak the upcoming sentence – essentially the expressive parameters it decided on. It might contain things like the chosen emotional tone or phoneme timing. This is advanced usage; typically, you can ignore this unless you want to visualize how EVI is modulating the speech.

* **AudioOutput (EVI → client):** This is the **audio of EVI’s speech** in small chunks . As EVI speaks, it streams the audio to you as a series of audio\_output messages. Each contains a portion of WAV/PCM data (likely base64 or binary WebSocket frame). The client should play these in sequence with minimal buffering to achieve real-time playback of the AI’s voice. By the time the assistant finishes speaking, you will have received multiple AudioOutput chunks totaling the full reply.

* **UserMessage (EVI → client):** This represents the **transcription of the user’s speech**. If verbose\_transcription=true was set in the query, you will get interim transcripts as UserMessage with an interim=true flag while the user is speaking . Once the user finishes, a final UserMessage (interim=false) is sent containing the final transcript text . This is useful if you want to display what EVI thinks the user said in real-time.

* **ChatMetadata (EVI → client):** Upon connection, EVI sends a metadata message with info like a chat\_id and chat\_group\_id for this session . The chat\_group\_id is the persistent conversation identifier – save this if you want to later resume context after a disconnect . The metadata may also include the config used, timestamps, etc.

* **AssistantEnd (EVI → client):** Indicates the assistant has finished speaking (end of its turn) . After you receive this, EVI expects the user to speak next (or the client can send a new query). The client could use this signal to re-enable the user’s microphone, for example.

* **ToolCall / ToolResponse (EVI ↔ client):** If you configured custom **tools** (essentially functions that EVI’s LLM can call for additional info, e.g., “lookupWeather”), EVI might send a ToolCallMessage with details of the requested operation . Your client should then execute the tool (e.g., fetch weather) and **respond** with a ToolResponseMessage containing the result . If a tool fails, a ToolErrorMessage can be sent by client . Tools are a powerful extension mechanism to give EVI dynamic knowledge/actions beyond the base conversation.

* **PauseAssistant / ResumeAssistant (client → EVI):** You can send these to programmatically pause EVI’s speech or resume it. For instance, if the user presses a “pause” button while EVI is speaking, you’d send a {"type": "pause\_assistant"} message to stop output (EVI will stop mid-utterance). You can later send resume\_assistant to continue. This ties into the interruptibility feature (though EVI auto-pauses on user interrupt as well).

* **Error (EVI → client):** If something goes wrong (e.g., malformed message), an Error message is sent. It contains an error\_code and message. Handle these gracefully – log them, display a message, or attempt recovery.

In summary, the WebSocket supports a rich protocol for full conversational control. However, you often don’t need to handle every type: a basic integration could simply send AudioInput and receive UserMessage (transcript) \+ AudioOutput (voice) and possibly AssistantMessage (text) for debugging. The Hume Python/JS SDKs wrap many of these details.

**Chat Lifecycle:** A typical EVI chat session flow is:

1. **Connect** to WebSocket with API key/token and config\_id.

2. Receive ChatMetadata (get chat\_group\_id).

3. **User speaks:** Client continuously sends AudioInput messages as user talks.

4. EVI sends interim UserMessages (if enabled) with partial transcript.

5. User stops – EVI sends final UserMessage (transcript).

6. EVI (LLM) processes for a moment, then streams out AssistantMessage (text of reply), AssistantProsody (optional info), and AudioOutput chunks (the reply spoken).

7. EVI sends AssistantEnd when done with the turn.

8. Loop back to user – EVI now listens for user (the client should resume sending audio).

9. Eventually, the session can **end** explicitly (client disconnects or sends a hang-up command, or user says goodbye and you choose to close). On close, you might receive a chat\_ended webhook event if configured .

**EVI Example – Python Streaming Conversation:** Below is a pseudocode example illustrating a simple EVI client using the Python SDK (for brevity). It captures microphone audio and plays back EVI’s audio response using simple libraries:

import asyncio  
from hume import AsyncHumeClient  
from hume.eventloop import start\_event\_loop  \# hypothetical event loop helper

API\_KEY \= "YOUR\_API\_KEY"  
config\_id \= "YOUR\_CONFIG\_ID"  \# EVI configuration with desired voice/persona

async def run\_evi\_chat():  
    client \= AsyncHumeClient(api\_key=API\_KEY)  
    \# Connect to EVI with a specific configuration  
    async with client.evi.chat.connect(config\_id=config\_id) as session:  
        print("Connected to EVI. You can start talking...")  
        async for event in session.stream\_microphone():  \# hypothetical method that handles mic I/O  
            if isinstance(event, str):  
                \# Assuming the SDK yields transcript text for interim/final user speech  
                print(f"User said: {event}")  
            elif event.get("audio\_output"):  
                \# Play the assistant's audio chunk  
                play\_audio(event\["audio\_output"\])  \# user-defined function to play raw audio bytes  
            elif event.get("error"):  
                print("EVI Error:", event\["error"\]\["message"\])  
                break

asyncio.run(run\_evi\_chat())

*Explanation:* We connect using an EVI config (assuming it’s set up with a prompt like “You are a helpful customer service agent,” and a friendly voice). We then continually stream microphone input and yield events. The example pseudocode treats events such that transcripts (UserMessages) come through as strings and audio chunks as a dict with "audio\_output". In practice, the Hume Python SDK likely provides higher-level abstractions, but under the hood it’s doing what we described: sending audio and receiving audio. As the user talks, interim transcripts print. When EVI speaks, we play the audio chunks. This loop could run indefinitely until the conversation is done.

**Configuring EVI:** Before using EVI, you’ll typically create a **Config** via Hume’s platform or API. In a config, you can set:

* **Initial Prompt:** A system or role prompt that defines the persona or goal of the AI. For example, *“You are an AI career coach. Speak in a calm, supportive manner. Your goal is to help the user reflect on their professional development.”* This guides the LLM’s responses.

* **Voice Selection:** Choose one of Hume’s pre-designed voices (identified by name or ID) or a custom voice for EVI’s speech. The voice defines the base sound (accent, age, timbre). EVI will still modulate it for emotional tone as needed. If not set, a default voice is used.

* **Language Model & API Keys:** Specify which LLM to use (Hume’s built-in model or an external one via API). Hume supports Anthropic, OpenAI, etc., so you would provide e.g. an OpenAI API key in the config if you want GPT-4 to handle the dialogue logic . You can also set maximum tokens, temperature, etc., for the LLM if needed.

* **Tools:** Register any custom tools that EVI can call. Each tool has a name and a specification (input/output schema). For example, a search() tool or a get\_current\_time() tool. In the config, you’d define how the LLM can invoke it. At runtime, when the LLM’s response triggers a tool use, your client gets the ToolCall message as described and must respond. Tools allow EVI to extend beyond its base knowledge.

* **Webhooks:** As mentioned, you can set webhook URLs and select which events to forward. Common events are chat\_started, chat\_ended for session tracking. The payload contains details like chat\_id, timestamps, config\_id, and possibly phone numbers if Twilio integration is used (more below) . There are also message-level webhooks (e.g. you could webhook each user message or each assistant message if you want to log entire conversations server-side in real time). Each event type has a specific JSON payload defined in Hume docs . Webhooks are optional but very useful for analytics or coordinating external actions (e.g., log conversation transcripts to a DB as they happen, or trigger an alarm if EVI detects the user is in crisis).

**EVI Rate Limits & Performance:** EVI is interactive, so traditional “requests per minute” limits don’t directly apply like an HTTP API. Instead, Hume might limit the **number of concurrent WebSocket sessions** you can have (e.g. certain account tiers may allow N simultaneous EVI sessions). Check your Hume plan for specifics. Each session can last a certain duration (if you keep it open for hours there might be an inactivity timeout – the docs imply inactivity or very long sessions will be closed with code 1000\) . In practice, if you need EVI with dozens of simultaneous users, ensure your account supports it or talk to Hume about scaling limits.

**Resuming Conversations:** If a user returns later or gets disconnected, EVI can **resume context** so it remembers past discussion. This is done via the chat\_group\_id. Every chat session has a chat\_group\_id (which by default is the same as chat\_id for new sessions). If you reconnect and provide resumed\_chat\_group\_id=\<ID\> in the WS query params, EVI will load the memory of that group . You can obtain the chat\_group\_id from the initial ChatMetadata message or via the list chats API. This feature ensures persistence across sessions (for example, a user’s phone call that drops can be re-established without starting over). You can also explicitly end a chat group if you want to clear context.

**EVI Error Handling & Ethics:** EVI may occasionally generate content that is inappropriate or incorrect (as with any LLM-based system). Hume likely has safeguards, and your config prompt can include instructions to refuse certain requests or avoid certain styles. Always test your specific config. If the user speaks in a language EVI doesn’t support (currently best in English; partial support for other languages), transcription may fail – you might get a Transcript confidence below threshold error if ASR can’t transcribe (this is more relevant to expression API, but EVI uses similar ASR) . In such cases, EVI might not respond meaningfully. Consider restricting advertised languages or handling that error by having EVI say “I’m sorry, I didn’t catch that.”.

**Closing the Session:** Either side can close the WebSocket to end the chat. On normal closure, you’ll get a chat\_ended webhook if configured . Hume might also close the socket if idle (no audio) for some time. If your client closes intentionally (user hits end call), you might want to send a final message or update UI accordingly.

### **Deployment Considerations for EVI**

Running EVI in production (e.g. as a voice assistant or call center agent) involves considerations beyond the API calls:

* **Audio Handling:** Use reliable methods to capture microphone or telephone audio and stream it to EVI with low latency. For web apps, use the Web Audio API or WebRTC getUserMedia. For phone calls, see Twilio integration below. Ensure audio is single-channel, appropriate sampling rate (8k for phone, 24k for web) to match EVI’s expectations (EVI will handle downsampling from 24k to its model’s rate).

* **Playback:** For the assistant’s voice output, you’ll need a way to play audio in real-time (e.g. an audio buffer in the browser, or if on a phone call, send it back through Twilio). Manage barge-in: if user speaks over EVI, you should stop playback immediately. EVI already stops generating, but your playback system needs to halt.

* **Scaling:** Each EVI session is relatively heavy (it runs an LLM and TTS in the loop). To scale to many users, consider running multiple instances or ensure your account’s throughput can handle it. Hume’s cloud will handle the heavy compute, but your server must handle multiple WebSocket connections, and possibly bridging audio. Measure bandwidth: audio streaming (PCM) can be \~? 128 kbps; multiply by users. Plan infrastructure (use async IO or threading as needed).

* **Monitoring:** Log important events – e.g., start/end of chats, any errors, conversation duration, user satisfaction if measurable. Use Hume’s webhooks to feed a monitoring service. Hume also provides a **Dashboard** and analytics on their platform you can check. The request\_id in responses can be used when contacting Hume support for issues .

* **Security:** If deploying a web client, use the token auth so the API key isn’t exposed. Also consider audio data sensitivity – use TLS (wss:// is secure). Hume is a processor of potentially personal data (voice can be identifying and content could be sensitive); review privacy requirements and perhaps inform users they are speaking with an AI (in fact, Hume’s terms and telephone regulations *require* informing users the voice is AI) . The **Telephone Consumer Protection Act (TCPA)** mandates consent for AI calls . Always follow these guidelines in telephony scenarios.

We will discuss specific integrations (Twilio, etc.) in a later section. Next, let’s cover the Text-to-Speech API, which is closely related to EVI’s output capability.

## **Octave Text-to-Speech (TTS) API – Expressive Voice Synthesis**

Octave is Hume’s cutting-edge **text-to-speech** system that generates spoken audio from text with human-like expression and contextual understanding . Unlike conventional TTS that reads text verbatim, Octave is built on a **speech-language model (an LLM)** that actually comprehends the text’s meaning, allowing it to infuse the speech with appropriate emotion, tone, and style . You can think of it as ChatGPT but for speech: it not only produces what to say (if given a prompt beyond raw text), but also *how to say it*. Key capabilities include:

* **Context-Aware Prosody:** Octave adapts pitch, pacing, and emphasis based on the context of the words . For example, it might whisper a sentence that is parenthetical or clearly an aside, or inject excitement into an exclamation. It “knows” when a sentence is a question vs. a statement, a happy statement vs a sad one, and alters delivery.

* **Emotion and Style Control:** You have fine control over the emotional expression of the synthesized voice. Through natural language **descriptions** (prompts) or **acting instructions**, you can direct Octave to speak with a certain emotion or style . For instance, you can request a line to be spoken *“in a tone of awe and quiet reverence”* or *“with angry intensity”*. The model will adjust the voice output accordingly. This enables highly dynamic storytelling, character voices, and nuanced performances from the AI.

* **Voice Design and Custom Voices:** Octave can produce virtually any kind of voice. You can either use one of the **100+ pre-designed voices** in Hume’s Voice Library or create your own. **Voice design** can be done by providing a descriptive prompt of a voice’s characteristics (e.g. *“a young adult female voice, warm and smoky, with a British accent”*), and Octave will generate a voice matching that description . Alternatively, **Voice Cloning** allows you to create a custom voice from a short audio sample (with consent) . Once you have a custom voice, you can use it via the API by specifying its ID. All voices (custom or library) are powered by Octave under the hood, so they support emotional modulation as well.

* **Multi-Utterance and Long-Form Support:** Octave is capable of handling long scripts (audiobook chapters, podcasts, dialogues). It preserves consistency of voice and emotion over long texts . You can provide **context** from earlier in the script to maintain tone (e.g., if previous paragraphs were sad, it keeps the sad tone unless it should shift). It can also manage multiple **characters** by switching voices between utterances if you specify different voices for different lines.

* **High Audio Quality:** It outputs high fidelity audio (supported formats include WAV (PCM 16-bit, 48 kHz), MP3, or raw PCM) . The default sample rate is 48 kHz for clear sound .

The TTS API provides both **synchronous** request/response for quick text-to-speech conversion and **streaming** endpoints for low-latency audio generation.

### **TTS API Endpoints and Usage**

**Endpoint Overview:** All endpoints are under https://api.hume.ai/v0/tts. The main ones are:

* **POST /v0/tts (JSON or File):** Submit a synthesis request. You can either get the result as JSON (with audio base64) or directly as an audio file, depending on the Accept header.

* **POST /v0/tts/stream (Streamed):** Initiate a streaming TTS response. This comes in two flavors: one that streams chunks of audio in an HTTP response, and one that streams via server-sent events or websockets. The docs label these as “Streamed File” vs “Streamed JSON” endpoints .

* **GET /v0/voices (and related):** Manage voices. You can list available voices, create new ones (though voice creation might be more through the “Voice design” endpoints or the platform UI).

**Request Format (Synthesis):** The primary payload for a TTS request is a JSON object that includes:

* **utterances:** *\[Required\]* An array of one or more utterance objects . Each utterance object typically has:

  * text – the string of text to speak .

  * voice – (Optional) specify which voice to use for this utterance. This can often be a voice ID or name from the voice library. If omitted, and no voice specified elsewhere, Octave will *dynamically generate a new voice* that fits any description given (or a default voice if no description).

  * description – (Optional) a natural language description of *how* to deliver this utterance . This is effectively an *acting prompt* for the style, emotion, accent, etc., specifically for this piece of text. e.g., description: "with a tone of excitement and wonder, as if telling a favorite story to a child".

  * speed – (Optional) a relative speaking speed control (if supported, e.g., 1.0 normal, 0.8 slower, 1.2 faster).

  * trailing\_silence – (Optional) how much silence (in seconds) to append after this utterance’s audio (useful for pauses).

* **context:** *\[Optional\]* A context object providing additional text that is *not to be spoken* but gives context for style . You can supply previous dialogue lines or narrative context here. The context helps Octave maintain consistency, especially if instant\_mode (see below) is off and it’s doing a more thorough analysis of the whole input.

* **format:** *\[Optional\]* Specifies audio format for output . You can request mp3, wav, or pcm. If not given, default might be WAV. In JSON response mode, the audio is base64 (with an encoding.format field telling you which format it is) . If you hit the file endpoint with Accept: audio/\*, the format can dictate Content-Type of response.

* **num\_generations:** *\[Optional, 1-5\]* How many alternative takes to generate . If you specify \>1, Octave will produce multiple different audio outputs for the same input (varying voice details or style). This is useful for getting options (say, 3 variations of a line read). Each generation will be returned in the generations list. **Note:** Instant mode (low-latency) only supports 1 generation at a time .

* **split\_utterances:** *\[Optional, boolean\]* Defaults to true. This controls how the API segments the audio output relative to your input utterances . If true, each input utterance might be split into smaller audio chunks in the response, at natural breakpoints (like sentences) – the response JSON will then include a nested snippets array for fine-grained alignment . If false, it will maintain a 1-to-1 mapping (each utterance yields one contiguous audio chunk).

* **instant\_mode:** *\[Optional, boolean\]* Defaults to true . This toggles **ultra-low latency streaming**. When instant\_mode:true, Octave will start streaming back audio almost immediately, rather than waiting to process the entire text. This mode is recommended for real-time applications (like EVI or an interactive web app) because it dramatically reduces the initial response delay . The trade-offs: you *must* specify a predefined voice (no dynamic voice creation) , and num\_generations must be 1 . Also, certain parameters like context or very complex prompts might be handled slightly differently (instant mode focuses on speed). If you want to generate a brand new voice from description, you’d disable instant\_mode for that request .

A typical simple JSON request could be:

{  
  "utterances": \[  
    {  
      "text": "Hello, welcome to our service.",  
      "voice": "pixie",   
      "description": "a friendly and cheerful tone"  
    }  
  \],  
  "format": {"format": "wav"},  
  "instant\_mode": true  
}

This asks Octave to say the text in the preset voice “pixie” (just an example name) in a friendly tone, quickly. It requests WAV audio and low latency.

**Response (JSON mode):** If you use the JSON endpoint (or set Accept: application/json), the response will be a JSON containing the generated audio as base64 and metadata. For example:

{  
  "generations": \[  
    {  
      "audio": "//PExAA0DDYRvkpNfhv3JI5JZ...etc.",  
      "duration": 7.44225,  
      "encoding": { "format": "mp3", "sample\_rate": 48000 },  
      "file\_size": 120192,  
      "generation\_id": "795c949a-1510-4a80-9646-7d0863b023ab",  
      "snippets": \[  
        \[  
          {  
            "audio": "//PExAA0DDYRvk... (base64)...",  
            "generation\_id": "795c949a-...same as above...",  
            "id": "37b1b1b1-...some snippet id...",  
            "text": "Hello, welcome to our service.",  
            "utterance\_index": 0  
          }  
        \]  
      \]  
    }  
  \],  
  "request\_id": "66e01f90-4501-4aa0-bbaf-74f45dc15aa7"  
}

This example (from docs) shows a single generation (since num\_generations:1). The audio field is a base64 string of the entire audio file for that generation. It also provided an encoding (mp3 at 48kHz) and a duration in seconds, plus the file\_size in bytes. The generation\_id is a UUID for this audio – you can use it to reference this generation (e.g., in MCP integration to replay audio by ID) . The snippets array here indicates that the utterance was split (maybe not really in this case since only one snippet with same text). If you had multiple utterances in input, or if an utterance was long and got split, snippets would contain subarrays of segments with their own audio chunks and references to which utterance and text segment they correspond to. This is useful for synchronizing audio to text (like highlighting words as spoken).

If you used the **File endpoint** (e.g., POST /v0/tts with Accept: audio/wav), the response would be directly the binary audio bytes (WAV data) with appropriate Content-Type: audio/wav. In that case you wouldn’t get JSON; your HTTP client would get the audio file which you can save or stream. Use that approach if you just need the audio result and not the extra metadata.

**Streaming TTS:** For real-time applications where even a second of latency matters (e.g., generating a long response on the fly), Hume offers streaming endpoints:

* **POST /v0/tts/stream/file:** This initiates a request and streams back the audio as it’s being generated, presumably using chunked transfer encoding. The audio bytes will start coming in as soon as available (Octave doesn’t wait to complete the whole sentence before starting output if instant\_mode is on).

* **STREAM /v0/tts/stream/json:** The docs indicate a “STREAM” method for JSON , likely meaning a WebSocket or Server-Sent Events channel where you get JSON messages (perhaps containing base64 chunks or progress). However, typically streaming audio is easier via binary chunks.

For most cases, instant\_mode:true plus the normal endpoints might suffice because that already yields quick first bytes. But if you want to play audio *as it’s generated* without holding the connection open yourself, the streamed file endpoint is convenient. For example, in Node or Python, you could hit the stream endpoint and start piping the response to a speaker as data arrives.

**Instant Mode:** Let’s expand on **Instant Mode** . With instant\_mode=true, Octave will deliver the first chunk of audio significantly faster (latency can drop to a few hundred milliseconds). It essentially means Octave starts generating and outputting audio before it has processed the entire text. It might use a heuristic to decide how to start speaking, then adjust as more text is read. The trade-off is you cannot use it when designing a brand new voice on the fly (because generating a new voice takes a brief moment that would add latency) – thus you must specify a voice (either custom or one from the library) when instant mode is on . Also, if you wanted multiple alternatives (num\_generations\>1), instant mode can’t do that concurrently . Use instant mode for interactive scenarios. Turn it off for batch TTS or when you need the highest fidelity/consistency in a single file output. The documentation suggests keeping instant mode *enabled* for Agent/LiveKit sessions where latency is key, and disabling it for voice design workflows or multi-generation requests .

**TTS API Limits:** The platform imposes some limits to keep in mind :

* **Rate limit:** 100 TTS requests per minute . Each request could be one or multiple utterances. If you exceed this, you’ll get HTTP 429 or similar. If you need higher throughput (like generating thousands of clips), you might need to throttle or contact Hume for higher limits.

* **Text length:** Max 5,000 characters per request . If you have a longer text (e.g., a book chapter), you should split it into multiple requests or utterances. The Project/Audiobook tool on Hume’s platform likely does such splitting automatically. 5k chars is roughly \~5-8 minutes of speech.

* **Description length:** Max 1,000 characters for the voice description or acting instruction field . This is plenty for most use (you usually need a sentence or two to describe style).

* **Max generations:** 5 per request . If you need more than 5 variants, you’d have to issue multiple requests perhaps.

* **Supported audio formats:** PCM (likely in WAV), MP3 are supported for output . If you need others like OGG, you might have to convert externally. PCM is lossless, MP3 is compressed.

If you hit these limits or provide invalid input, you’ll get error codes. E.g., if text is too long, a 400 error with message. Common errors might include voices not found (if you specify a voice id incorrectly), or authentication issues.

### **Advanced TTS Features**

**Voice Selection & Library:** Hume provides a large voice library out-of-the-box . These include a range of accents, ages, and styles (often given human names or descriptive titles). You can query available voices via API (likely GET /v0/voices). Each voice has an id and name, and possibly metadata (gender, description). To use a specific voice in TTS, you include "voice": {"id": "voice\_id\_string"} or sometimes just "voice": "Name" if the API accepts name. The docs show using VoiceByName("John") in their LiveKit example for selecting a voice .

If no voice is specified, Octave will dynamically create one based on the description prompt. For example, if you provided a description like *“a deep, gravelly old man’s voice”* and no voice, it will generate a novel voice with those qualities on the fly. This is powerful but slower (hence not allowed in instant mode). If you like a dynamically generated voice and want to reuse it, you can actually **save it as a custom voice** via the API or Hume console (there’s a save\_voice tool in MCP server as we saw) . This gives it an ID for future use.

**Acting Instructions vs. Voice Description:** These two might seem similar but serve slightly different purposes. The *voice description/prompt* (used when designing or selecting a voice) describes the general persona or qualities of the voice. The *acting instructions* (which we pass in description for an utterance) are more about how to deliver the specific line. For example, your voice might be “Narrator: middle-aged calm British male.” But acting instruction: “whisper this line as if revealing a secret.” You can combine them: voice gives baseline, instructions fine-tune performance on each utterance.

Hume docs differentiate these in the UI as well (there’s a **Prompting guide** and **Acting instructions** doc) . Essentially, use the voice design prompt to set up the voice, and use acting instructions to adapt emotion dynamically in content.

**Continuation & Segmentation:** If generating a long piece, the API might split it (especially if split\_utterances=true). For example, if you input a whole paragraph as one utterance, Hume might return multiple snippets corresponding to sentences. This segmentation ensures natural intonation and breathing points. If you want a single continuous audio file, you can either set split\_utterances:false (then one utterance in equals one chunk out) , or you can concatenate the snippets on your end. The JSON gives you snippet IDs and text, so you could reconstruct or align if needed. For interactive scenarios, leaving it true is fine because you often play as it comes.

**Multi-Speaker Dialogue:** You can submit multiple utterances with different voices in one request. E.g.:

{  
 "utterances": \[  
   {"text": "Alice: Hi\! How are you?", "voice": "AliceVoice"},  
   {"text": "Bob: I'm doing well, thanks.", "voice": "BobVoice"}  
 \]  
}

Octave will produce two audio segments, one for Alice line in AliceVoice, one for Bob line in BobVoice, possibly concatenated (depending on split settings). If you set split\_utterances=true (default), you’d get snippets grouping each utterance anyway. This is useful for generating dialogues or multi-character audiobooks. Hume’s system will preserve distinct speaker styles.

**Real-Time vs Batch Use:** If you just need to generate static audio files (e.g., pre-recorded messages, entire audiobook chapters), you can use the normal POST and wait for the response (since 5000 chars limit, split chapters). For real-time (like responding to user input on a server), consider streaming or instant mode to start playback sooner. For example, if a chatbot wants to speak out each answer as it’s ready, you can call streaming TTS with the answer text and start streaming audio to the client.

### **Python Code Examples for TTS**

**Example 1: Simple TTS (REST, single utterance)** – converting a text to speech and saving to a file:

import requests  
API\_KEY \= "YOUR\_API\_KEY"  
text \= "Thank you for calling. Your feedback is valuable to us."  
voice\_id \= "expert\_female\_en\_US"  \# assume a voice ID from library  
payload \= {  
  "utterances": \[  
    {  
      "text": text,  
      "voice": {"id": voice\_id},  
      "description": "polite and helpful tone"  
    }  
  \],  
  "format": {"format": "wav"}  
}  
resp \= requests.post("https://api.hume.ai/v0/tts", json=payload,  
                    headers={"X-Hume-Api-Key": API\_KEY, "Accept": "audio/wav"})  
resp.raise\_for\_status()  
with open("output.wav", "wb") as f:  
    f.write(resp.content)  
print("Audio saved to output.wav (length {:.1f}s)".format(len(resp.content)/32000))

Here we directly accept audio, so resp.content contains WAV data. We used a library voice and a description. The length estimate at the end is just rough (assuming 32kB per second for 16kHz mono WAV for estimation). In practice, use the duration from JSON if needed.

**Example 2: Streaming TTS (low latency)** – using WebSocket (hypothetical, Hume might allow TTS via WS as well, or we simulate chunked reading):

import websocket, json, base64

API\_KEY \= "YOUR\_API\_KEY"  
ws \= websocket.create\_connection("wss://api.hume.ai/v0/tts/stream/json?api\_key="+API\_KEY)  
\# Prepare request for streaming JSON endpoint  
req \= {  
  "utterances": \[  
    {"text": "This is a real-time generated response.", "voice": {"name": "Eleanor"}}  
  \],  
  "format": {"format": "pcm"},  
  "instant\_mode": true  
}  
ws.send(json.dumps(req))  
audio\_data \= b""  
while True:  
    msg \= ws.recv()  
    if not msg:  
        break  
    data \= json.loads(msg)  
    if "audio\_chunk" in data:  
        audio\_data \+= base64.b64decode(data\["audio\_chunk"\])  
        play\_buffer\_nonblocking(audio\_data)  \# pseudo-function to play as it streams  
    if data.get("is\_last\_chunk"):  
        break  
ws.close()

*Note:* The above is speculative – the actual streaming JSON might wrap audio in an “audio” field. The idea is that the TTS WS sends chunks with a flag when done. We accumulate and can play incrementally.

**Example 3: Using Hume’s Python SDK for TTS:**

from hume import HumeClient  
from hume.tts import Voice, FormatWav

client \= HumeClient(API\_KEY)  
voice \= Voice(name="Eleanor")  \# using a preset by name  
resp \= client.tts.synthesize(text="Hello world\!", voice=voice, format=FormatWav())  
audio\_bytes \= base64.b64decode(resp\["generations"\]\[0\]\["audio"\])  
with open("hello.wav", "wb") as f:  
    f.write(audio\_bytes)

This abstracts things. The SDK likely also has convenience for streaming (maybe an async method or similar as seen in their docs for SDKs ).

**Error Handling:** Common errors in TTS might include:

* *400 Bad Request:* e.g., if required fields missing, or if text is empty. The error will say “Unprocessable Entity” if the JSON is malformed .

* If you put a voice name that doesn’t exist, likely an error like “voice not found”.

* If your API quota is exceeded or API key invalid, you’d get 401/403.

* If using token auth, ensure to pass Authorization: Bearer \<token\> for REST calls .

### **Integration & Deployment for TTS**

Using TTS in production involves deciding whether you need real-time streaming or if you can pre-generate audio:

* **Real-Time Generation:** e.g. an on-demand assistant or dialog system. Here you’d likely use instant\_mode and possibly streaming to minimize delay. If using in a user-facing app, hide the generation time with a loading spinner or generate progressive audio.

* **Pre-Generation (Batch):** e.g., generating an audiobook or voice-over. Hume’s **Projects** interface (and possibly API) lets you manage long-form synthesis by splitting text and generating in sequence . If doing via API, script it: break text into \<=5000 char chunks (perhaps at paragraph boundaries), call TTS for each chunk (maybe with voice and context from previous chunk to maintain tone). Stitch results if needed. The audio quality will be high; just ensure consistent voice usage and provide context so each chunk flows.

* **Monitoring and Costs:** TTS usage is metered typically by characters or time. Hume’s pricing might give you N characters or seconds free, then charge beyond . Monitor how many characters you’re synthesizing (each request could return that in request\_id or so). If you exceed monthly quotas, Hume either stops or charges for extra (the FAQ mentions upgrade or pay per extra 1000 chars) .

* **Caching:** If you have phrases that get synthesized repeatedly, consider caching the audio results (store by input text \+ voice \+ style). Instead of hitting API every time for common phrases, reuse the cached audio. E.g., system prompts or frequently used responses.

* **Voice IP and Usage:** You can use Hume’s generated voices in commercial projects (with appropriate plan) . Hume’s terms grant you ownership of output audio but also a license for them to improve their models with it . Just be mindful of not violating any terms (e.g., don’t use voices to create defamatory content, etc., see prohibited uses).

* **Latency Considerations:** Usually TTS (especially with instant mode) is quite fast for short inputs (a few hundred ms to start). For very large inputs (multi-thousand chars), it could take several seconds. If real-time, break the text up so you can start playing first part while latter part generates. Essentially what streaming does.

* **Twilio/Phone usage:** If you want to play TTS over a phone call (outside EVI context), Twilio can play audio files or you could stream via TwiML . Alternatively, generate the sentence audio via TTS and give Twilio the URL to an MP3 to play. That’s slower (needs generation then Twilio fetch). If truly interactive, you might integrate directly as EVI (which uses TTS internally).

* **Client-Side TTS vs Server:** If building a web app and concerned about latency, note that Hume’s TTS requires server calls (no client-side running). But you can call it from the browser if you trust the environment (with a token). The Vercel AI SDK integration (next section) actually allows using Hume TTS easily in a Next.js app.

## **Integration Options and Advanced Integrations**

Hume AI’s platform is designed to integrate with other technologies and services in the AI ecosystem. Here we discuss optional but powerful integrations with Twilio for telephony, LiveKit for real-time video/audio, Vercel’s AI SDK for web apps, and MCP (Model Context Protocol) for connecting Hume’s TTS with AI assistants like Claude.

### **Twilio (Telephone Calls with EVI)**

**Use Case:** Connecting Hume’s EVI to a phone call – e.g., having a user call a phone number and talk to the AI, or the AI calling out to a user. Twilio is a popular cloud communications platform that can route voice calls and integrate with APIs.

Hume offers a direct Twilio integration endpoint to simplify inbound calls: **POST /v0/evi/twilio** . This endpoint bridges Twilio’s call audio to EVI without you running a server. The setup is:

* **Inbound Calls:** You buy a number on Twilio and set its “Voice URL” webhook to https://api.hume.ai/v0/evi/twilio?config\_id=\<YOUR\_CONFIG\_ID\>\&api\_key=\<YOUR\_API\_KEY\> . When someone calls your number, Twilio will start a stream to Hume’s endpoint, and EVI will handle the call. EVI uses the config\_id you provide (so set up a config with the desired behavior). The caller and EVI can converse in real time. All of EVI’s core features (emotion detection, etc.) work over the phone . *Limitations:* Because this direct integration bypasses your server, you **cannot inject custom tools, context, or control pause/ resume** via the API during the call . It’s a straight user \<-\> EVI conversation. If you need those advanced controls, you’d need to handle the audio bridging yourself (see next point).

* **Custom Bridge (Advanced):** Alternatively, you can use Twilio’s  or  WebSockets from TwiML to pipe audio to your own server, then forward to EVI. Twilio can send raw audio to a WS URL you specify and accept audio in return to play to caller. In this setup, you’d have a server that connects to both Twilio’s audio stream and Hume’s EVI WS, shuttling audio back and forth. This is complex but allows you to also intercept messages (like Tool calls, etc.). Hume’s docs suggest doing this if you need tools or more control .

* **Outbound Calls:** If your AI needs to call users (outbound), you would initiate a call via Twilio’s API and similarly use the Hume bridging. Hume supports outbound through Twilio, but importantly **you must have the user’s explicit written consent** due to FCC rules (automated outbound AI calls require opt-in) . To do this, you’d likely use Twilio’s API to create a call that connects to the same evi/twilio endpoint or your server bridge. Ensure you’ve logged consent – violations can incur heavy fines .

**Quality and Latency:** When using telephony, note audio is 8 kHz mono (telephone standard). EVI will work with that, but the quality is lower, and some nuance may be lost. Also, Twilio’s routing adds a few hundred milliseconds latency in each direction . Hume warns that interactions will be slightly slower than web because of this (\~200-300ms extra) . There’s not much to do except be aware. EVI’s end-of-turn detection still works, but might have a tiny delay. The user experience is still acceptable – just not as snappy as say talking to EVI on a web app.

To configure Twilio integration, Hume’s guide steps are :

1. In Twilio Console, buy a number.

2. Create an EVI configuration on Hume (get the config\_id).

3. Set the number’s voice webhook URL to https://api.hume.ai/v0/evi/twilio?config\_id=YOUR\_CONFIG\_ID\&api\_key=YOUR\_API\_KEY.

4. Call the number to test – you should reach the EVI agent.

Twilio will handle converting the call audio to the stream EVI needs. No coding required for inbound aside from config setup. For logging calls, you can rely on EVI’s webhooks (e.g., log chat\_started with caller number from payload ). Twilio’s webhook will include the caller’s number in the caller\_number field of chat\_started event , letting you map the conversation to that user. If needed, you could use that to pick a certain persona (though right now the config\_id is fixed in URL, you’d need multiple numbers or a smarter proxy to vary config per caller).

In summary, Twilio integration makes it easy to give EVI a phone interface, enabling voice bots that people can call as if they were real call center agents or personal assistants.

### **LiveKit (Interactive Audio/Video via LiveKit Agents)**

**Use Case:** Integrating Hume’s TTS (and possibly EVI) into real-time WebRTC sessions such as video conferences or interactive audio rooms. **LiveKit** is an open-source platform for low-latency audio/video, similar to Zoom or WebRTC SFU. LiveKit introduced an **Agents** framework that allows running AI modules in the call pipeline (for transcription, bots, etc.) . Hume provides a plugin for LiveKit Agents to use its TTS within those pipelines .

In practical terms, imagine a LiveKit room where one participant is an AI that speaks using Hume’s TTS. The LiveKit Hume TTS plugin can generate speech in response to events in the room (e.g., after an STT \+ LLM processes what a user said).

**Integration modes:**

LiveKit Agents supports two modes as per Hume docs :

* **AgentSession:** a full pipeline with STT \-\> LLM \-\> TTS. In this mode, you instantiate an Agent that uses Hume TTS for output. This is similar to EVI but you can mix and match STT/LLM providers. You’d use AgentSession if you want a conversational agent in a LiveKit call. Hume’s advice: enable instant\_mode for low latency, and **specify a voice** up front for the session (so it’s consistent and because instant mode needs a fixed voice) . They also say *don’t* use per-request crafting parameters (like context or description) in AgentSession – pick a voice and stick with it for all responses in that session to maintain continuity .

* **Standalone TTS:** using Hume TTS as a standalone service in LiveKit (not necessarily tied to an LLM agent). This could be if you just want to synthesize some announcements or convert text to speech on the fly within the call. In standalone mode, you can set voice and style per request freely .

**Setup:** To use Hume TTS in LiveKit Agents, you’ll need to install the Hume plugin package (@livekit/plugins-hume for Node, or similar) and provide your Hume API key and LiveKit server credentials . The Hume docs snippet shows:

* Set environment: HUME\_API\_KEY, LIVEKIT\_URL, LIVEKIT\_API\_KEY, LIVEKIT\_API\_SECRET .

* In code, import the Hume TTS plugin and use it in your agent pipeline. For example (from docs) :

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions  
from livekit.plugins.hume import TTS, VoiceByName, VoiceProvider

class VoiceAssistant(Agent):  
    def \_\_init\_\_(self):  
        super().\_\_init\_\_(instructions="Your system prompt...")  
        \# The Agent base might use instructions as the context for LLM.

    async def on\_audio(self, audio\_stream, context: JobContext):  
        \# Process incoming audio with STT, LLM, etc. (simplified)  
        user\_text \= transcribe(audio\_stream)  
        response\_text \= my\_llm.generate(user\_text)  
        \# Use Hume TTS to speak response:  
        tts \= TTS(voice=VoiceByName("Eleanor"), provider=VoiceProvider.HUME)  
        audio\_out \= await tts.synthesize(response\_text)  
        return audio\_out

\# Then set up LiveKit AgentSession to use this VoiceAssistant, etc.  
session \= AgentSession.create(options=WorkerOptions(auto\_start=True))  
session.add\_agent(VoiceAssistant())  
session.start()

This pseudo-flow shows how you might call the Hume TTS plugin (TTS(...)) asynchronously to get audio. The plugin handles calling Hume’s API (using the HUME\_API\_KEY you set). The VoiceByName is used to pick the voice. The snippet also highlights enabling instant\_mode: They mention enabling it for responsive performance . Possibly the TTS plugin has an option like TTS(..., instant\_mode=True) or it defaults to true if you provided a voice.

**Constraints:** They note some constraints – in AgentSession, because it persists, you should:

* Use one voice throughout (done by specifying in TTS construction once).

* Not vary certain parameters per call (like you wouldn’t give a new description each time, because AgentSession likely uses the initial voice config for speed).

If you need to occasionally change style, you might consider splitting into separate sessions or using standalone mode for that utterance.

**Resources:** Hume provides a complete example project on GitHub for LiveKit integration , which is a great starting point. It likely includes code to handle the audio streams and integrate with some STT (maybe AssemblyAI or Vosk) and an LLM (maybe OpenAI) since Hume doesn’t provide STT in this plugin (though Hume’s expression API could do ASR but not sure if the plugin uses that).

In short, LiveKit integration is geared toward developers building real-time experiences (like an AI participant in a video call or a live audio room) and want Hume’s expressive voices to speak. It leverages the performance of Hume’s streaming TTS in a WebRTC environment.

### **Vercel AI SDK (Next.js Web Apps)**

**Use Case:** Integrating Hume TTS into a web application, particularly a Next.js or similar frontend, using the Vercel AI SDK. The **Vercel AI SDK** provides React hooks and utilities to call AI models (usually ChatGPT, etc.) from the browser or edge functions easily . Hume has created a provider that plugs into this SDK to simplify using Hume’s speech API on the frontend .

**Installation:** As per Hume docs :

1. Install the core ai SDK (npm install ai) and Hume provider package (npm install @ai-sdk/hume).

2. Set your Hume API key in an environment variable (e.g., .env.local with HUME\_API\_KEY=...) .

3. Configure the Hume provider in your app. For example, in a Next.js project, you might have a file that initializes the provider:

// e.g., lib/hume.js  
import { createHume } from "@ai-sdk/hume";  
export const hume \= createHume({ apiKey: process.env.HUME\_API\_KEY });

This createHume registers Hume’s endpoints with the Vercel AI system using your key .

**Usage in Frontend:** The AI SDK exposes a generateSpeech function (or a React hook) to call TTS. Hume’s integration uses hume.speech() as the model identifier . For example, in a Next.js page or React component, you could do:

import { experimental\_generateSpeech as generateSpeech } from 'ai';  
import { hume } from '../lib/hume';

async function speak(text) {  
  const result \= await generateSpeech({  
    model: hume.speech(),   
    text: "Hello world\!"  
  });  
  const audioSrc \= result.audioUrl; // The SDK might give an audio URL or blob  
  new Audio(audioSrc).play();  
}

The SDK likely returns an object with either a direct URL to a stream of the audio or the audio bytes. Possibly it streams it under the hood and provides an \<audio\> element source when done. The docs snippet indicates usage .

**Advanced usage:** You can pass additional options to the generateSpeech call, such as specifying a voice or instructions:

* To **specify a voice**: Provide voice: "\<VoiceName or ID\>" in the generateSpeech call. The Hume provider will include that. E.g., generateSpeech({ model: hume.speech(), text: "...", voice: "Eleanor" }); would choose that voice .

* To **add instructions**: Likely an instructions or similar field. The doc suggests “Add instructions” as heading . Possibly usage: generateSpeech({ model: hume.speech(), text, voice: id, instructions: "angry tone" }).

* To **provide context**: The integration might allow a context parameter if you want to maintain style across multiple calls (less crucial in a single page app scenario unless generating long text in parts).

Under the hood, the Vercel AI SDK will call an Edge Function or API route that communicates with Hume’s API using your API key securely (it doesn’t expose it to the client – likely the call goes through Vercel servers). The result is then streamed to the client. This simplifies frontend integration as you don’t have to manually write fetch calls or handle base64.

The example next.js project likely shows how to set up an endpoint (maybe /api/generateSpeech) that uses Hume’s API. Possibly the createHume function abstracts that for you.

This integration is “optional” but very handy if you’re building a voice app in Next.js. It’s mostly focused on TTS usage (not the full EVI). For full EVI on web, you’d manage the WebSocket and audio capture yourself; though Hume has a React SDK too (hume-react-sdk) which might wrap some of that.

### **MCP (Model-Context Protocol) and Claude/Cursor Integration**

**Use Case:** Using Hume’s TTS in conjunction with AI assistants or IDE tools that support the **Model Context Protocol (MCP)**. MCP is a standard that allows LLM clients to use external tools via a unified interface . For example, **Cursor** (an AI coding assistant IDE), **Claude Desktop**, and **Windsurf** (an AI UX tool) support MCP to interact with tools.

Hume provides a **MCP Server** that implements the MCP interface for Hume’s TTS . By running this local server (via npx @humeai/mcp-server), you can empower these AI assistants to generate speech using Hume’s voices as one of their “tools.” Essentially, the AI (like Claude) can orchestrate text-to-speech requests through MCP calls.

From Hume’s docs , the Hume MCP server exposes tools like:

* tts – to synthesize and optionally play speech from text. (The assistant can call this to speak responses.)

* play\_previous\_audio – to replay a prior generation by ID, etc. (For instance, to compare takes.)

* list\_voices – to get available voices (so the assistant can decide or let you choose).

* save\_voice / delete\_voice – to manage custom voices in the library.

These allow quite sophisticated control. For example, you could have an AI in Cursor that when you ask, “Read this code snippet aloud”, it uses the tts tool to speak it. Or in Windsurf (which might be a design tool), you can have characters speak using Hume voices. The **Claude Desktop integration** presumably means you can have a conversation with Claude and ask it to speak the reply out loud via Hume TTS.

**Quickstart MCP:** As docs show , configuration involves adding the Hume MCP server as a tool provider in your MCP-compatible client’s config (e.g., adding it to Cursor’s .cursor/mcp.json). You supply your Hume API key as an env var for that process. Once running, the assistant recognizes tool names like “hume: tts” or similar.

**Example:** In Cursor’s config (from docs) :

"mcpServers": {  
    "hume": {  
        "command": "npx",  
        "args": \["@humeai/mcp-server"\],  
        "env": { "HUME\_API\_KEY": "\<your\_hume\_api\_key\>" }  
    }  
}

Then in Cursor, you could type something like /use hume.tts "Hello world" as a command (if that’s how it triggers tools), and it would synthesize speech.

The **benefit of MCP** is that it lets AI assistants decide *when* to use speech. For instance, Claude could decide based on context if a response should be spoken aloud (via TTS) or just written. It adds a dynamic speech capability to text-based AI.

This is especially useful for voice prototyping: you can work with a chat AI to design a character voice via prompts and have it generate and refine using Hume’s TTS – e.g., “Try that line with more sarcasm” and the AI can call tts with the updated instruction.

**System Requirements:** The MCP server requires Node.js and optionally an audio player (if you want it to automatically play sound on your local machine) . It suggests using ffplay for playback, but in a headless scenario you might not use playback, just generation (the audio data goes to the MCP client which might handle playback itself).

**Security:** The MCP server uses your API key and will consume your Hume credits as if you were calling directly . Only run it in a trusted environment.

In summary, MCP integration is a developer convenience to hook Hume TTS into existing AI assistant workflows (particularly for those using tools like Cursor or Claude). It’s optional but can accelerate workflows like voice content creation, multi-modal AI agent development, etc.

---

## **Authentication, Authorization & Deployment Best Practices**

Before deploying any Hume AI solution, ensure you handle **authentication** properly and consider **scalability, monitoring, and data management**:

**API Keys & Access Tokens:** Each Hume account has an API Key and Secret . The simplest auth is using the API Key directly (in X-Hume-Api-Key header for REST, or ?api\_key= for WS) . This is fine for server-side use. **Never expose the secret or API key in client-side code** (browser or app). For client-side (e.g., direct from a web page), use the **Token strategy**: use your backend to call Hume’s token endpoint (POST /oauth2-cc/token with Basic auth using key:secret) to get a temporary access\_token . This token lasts 30 minutes . You then give it to the client, which can connect via ?access\_token= or Authorization: Bearer \<token\> . This way the API key stays hidden. Automate token refresh if needed (e.g., client asks backend for a new one when expired).

**Rate Limits & Quotas:** Keep track of usage relative to your plan. Expression API has generous file limits but if you plan on analyzing thousands of videos, consider request pacing. TTS has a clear 100 req/minute limit – if you do high-volume TTS (like generating hundreds of files quickly), implement a queue or throttle (e.g., process 50 per minute to stay safe). If you foresee hitting limits, contact Hume for higher volume arrangements or try to distribute calls across time.

**Error Handling & Retries:** Implement retry logic for transient errors (HTTP 500 or 502 from Hume, or 1011 WebSocket server error ). Exponential backoff for a couple retries is wise. For 429 rate-limit errors, backoff longer or reduce request rate. For explicit errors (like invalid input), log and fix the input rather than retrying. With streaming websockets, if disconnected, attempt to reconnect if appropriate: for EVI, use resumed\_chat\_group\_id to continue context . For expression stream, simply reconnect and resend last config if needed. Always handle Error messages from websockets gracefully – EVI will send errors if, say, the config\_id is invalid or a tool call failed.

**Logging & Monitoring:** Use Hume’s provided **webhooks** and status endpoints to monitor activity. For example, log every chat\_started and chat\_ended event (with timestamps, user IDs) to measure session counts and durations. Similarly, for batch jobs, perhaps log when jobs complete and how many items processed. The request\_id in TTS responses or job\_id in batch can correlate logs between your system and Hume’s. Hume also has a **Status** page and possibly a usage dashboard (the docs refer to a usage page for credit limits ). Check status.hume.ai for any service issues if your calls fail unexpectedly .

**Data Storage and Schema:** If you plan to store the results of Hume’s analysis in a database, design a schema that can handle the complexity:

* For **batch expression results**: The output for each input is rich (multiple models, time series). One approach is a relational model: e.g., a table for Media (id, metadata like filename, timestamp, person\_id, etc.), a table for EmotionScore (media\_id, time, model, emotion, score). But that could be millions of rows if long videos with scores every few ms. Alternatively, store the raw JSON in a NoSQL DB or blob store, and create summary fields for quick queries. For example, store aggregate metrics per call: average sentiment, max anger time, etc., in columns for easy retrieval, while keeping full detail in JSON for drill-down. Another approach: use a time-series DB (if high frequency) or vector store if using the embedding (Hume doesn’t directly give an embedding, but the scores vector is effectively an emotion embedding – you could use it for similarity).

* For **TTS outputs**: If saving audio files, have a storage plan (cloud storage with URLs, or store paths in DB). If generating on the fly each time, maybe caching isn’t needed. If user can choose voices, you might store their preference (just the voice ID).

* For **Chat transcripts**: EVI transcripts can be stored in conversation logs. A schema with ChatSession (id, user, start\_time, end\_time, etc.) and ChatMessage (id, session\_id, speaker (user/assistant), text, timestamp, maybe emotion metadata like user’s measured emotions for that utterance). Hume gives you the user’s expression per sentence, which you could attach to the message for analysis (e.g., mark that the user was “frustrated” in a particular question).

**Scaling & Performance Optimization:**

* Use asynchronous IO. For example, when submitting batch jobs, you don’t need to block – you can fire off jobs concurrently up to the 500 job limit. For streaming, use non-blocking frameworks (async/await) so one slow connection doesn’t stall others.

* If you have many concurrent users (say 100+ EVI sessions), ensure your server (if bridging audio or handling token gen) can handle many websocket connections and audio processing. Consider horizontal scaling: Hume’s cloud can scale on their side; your side might need multiple instances behind a load balancer, each managing some sessions. Use sticky sessions if websockets (or a service like socket.io with sticky).

* **Latency**: For minimal latency in EVI or streaming, deploy your server in a region close to Hume’s servers. Check if Hume is multi-region; if not documented, assume US. Also ensure the client’s audio path is short (e.g., a user in Asia connecting to a server in US East to talk to Hume – could cause latency; ideally pick region).

* **Websocket Keepalive**: Implement pings/pongs or use a library that does to keep connections alive. Hume might close after inactivity, but sending periodic pings can show connection is alive. The Hume error code said it closes after too long inactivity or max session length with 1000 code .

**Privacy & Compliance:** If analyzing personal user data (voice, video) with Hume, consider user consent and data handling. Hume’s privacy policy should be reviewed . If you store emotional data, treat it as sensitive (it can reveal mood, health, etc.). For phone applications, as mentioned, inform the user they’re speaking with AI (TCPA, etc.) . For any recorded data, ensure compliance with local laws (e.g., two-party consent states if calls are recorded for analysis).

**Conclusion:** Hume AI’s platform offers a comprehensive toolkit for adding emotional intelligence to applications. The Expression APIs give you deep insight into human behavior from raw data, while EVI and TTS enable creating interactive and expressive AI agents. By leveraging these with robust engineering practices – proper authentication, error handling, and thoughtful integration – you can build systems that are not only powerful and feature-rich but also reliable and secure. With the information and examples provided in this documentation, you should be fully equipped to implement any Hume AI use case, from analyzing sentiment in thousands of support calls to deploying a real-time empathetic voice assistant.

