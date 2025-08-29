[Expression Measurement APIBatch](https://dev.hume.ai/reference/expression-measurement-api/batch/list-jobs)  
**List jobs**  
**GET**  
https://api.hume.ai/v0/batch/jobs  
**GET**  
/v0/batch/jobs  
cURL

| $ | curl https://api.hume.ai \\ |
| :---- | :---- |
| \> |     \-H "X-Hume-Api-Key: \<apiKey\>" |

[Try it](https://dev.hume.ai/reference/expression-measurement-api/batch/list-jobs?explorer=true)  
200Retrieved

| 1 | \[ |
| :---- | :---- |
| 2 |  { |
| 3 |    "job\_id": "job\_id", |
| 4 |    "request": { |
| 5 |      "callback\_url": null, |
| 6 |      "files": \[ |
| 7 |        { |
| 8 |          "filename": "filename", |
| 9 |          "md5sum": "md5sum", |
| 10 |          "content\_type": "content\_type" |
| 11 |        } |
| 12 |      \], |
| 13 |      "models": { |
| 14 |        "burst": {}, |
| 15 |        "face": { |
| 16 |          "descriptions": null, |
| 17 |          "facs": null, |
| 18 |          "fps\_pred": 3, |
| 19 |          "identify\_faces": false, |
| 20 |          "min\_face\_size": 60, |
| 21 |          "prob\_threshold": 0.99, |
| 22 |          "save\_faces": false |
| 23 |        }, |
| 24 |        "facemesh": {}, |
| 25 |        "language": { |
| 26 |          "granularity": "word", |
| 27 |          "identify\_speakers": false, |
| 28 |          "sentiment": null, |
| 29 |          "toxicity": null |
| 30 |        }, |
| 31 |        "ner": { |
| 32 |          "identify\_speakers": false |
| 33 |        }, |
| 34 |        "prosody": { |
| 35 |          "granularity": "utterance", |
| 36 |          "identify\_speakers": false, |
| 37 |          "window": null |
| 38 |        } |
| 39 |      }, |
| 40 |      "notify": true, |
| 41 |      "text": \[\], |
| 42 |      "urls": \[ |
| 43 |        "https://hume-tutorials.s3.amazonaws.com/faces.zip" |
| 44 |      \] |
| 45 |    }, |
| 46 |    "state": { |
| 47 |      "created\_timestamp\_ms": 1712587158717, |
| 48 |      "ended\_timestamp\_ms": 1712587159274, |
| 49 |      "num\_errors": 0, |
| 50 |      "num\_predictions": 10, |
| 51 |      "started\_timestamp\_ms": 1712587158800, |
| 52 |      "status": "COMPLETED" |
| 53 |    }, |
| 54 |    "type": "INFERENCE" |
| 55 |  } |
| 56 | \] |

Sort and filter jobs.

### **Headers**

X-Hume-Api-KeystringRequired

### **Query parameters**

limitintegerOptionalDefaults to 50

The maximum number of jobs to include in the response.

statuslist of enumsOptional

Include only jobs of this status in the response. There are four possible statuses:

* QUEUED: The job has been received and is waiting to be processed.  
* IN\_PROGRESS: The job is currently being processed.  
* COMPLETED: The job has finished processing.  
* FAILED: The job encountered an error and could not be completed successfully.

Allowed values:QUEUEDIN\_PROGRESSCOMPLETEDFAILED

whenobjectOptional

Specify whether to include jobs created before or after a given timestamp\_ms.

timestamp\_mslongOptionalDefaults to 1704319392247

Provide a timestamp in milliseconds to filter jobs.

When combined with the when parameter, you can filter jobs before or after the given timestamp. Defaults to the current Unix timestamp if one is not provided.

sort\_byobjectOptional

Specify which timestamp to sort the jobs by.

* created: Sort jobs by the time of creation, indicated by created\_timestamp\_ms.  
* started: Sort jobs by the time processing started, indicated by started\_timestamp\_ms.  
* ended: Sort jobs by the time processing ended, indicated by ended\_timestamp\_ms.

directionobjectOptional

Specify the order in which to sort the jobs. Defaults to descending order.

* asc: Sort in ascending order (chronological, with the oldest records first).  
* desc: Sort in descending order (reverse-chronological, with the newest records first).

### **Response**

object

Show 4 properties

Was this page helpful?

YesNo

[Previous](https://dev.hume.ai/reference/empathic-voice-interface-evi/chat-groups/get-audio)

#### [**Start inference job**](https://dev.hume.ai/reference/expression-measurement-api/batch/start-inference-job)

[Next](https://dev.hume.ai/reference/expression-measurement-api/batch/start-inference-job)

[Built with](https://buildwithfern.com/?utm_campaign=buildWith&utm_medium=docs&utm_source=dev.hume.ai)

\*\*\*

**Start inference job**  
**POST**  
https://api.hume.ai/v0/batch/jobs  
**POST**  
/v0/batch/jobs  
cURL

| $ | curl \-X POST https://api.hume.ai/v0/batch/jobs \\ |
| :---- | :---- |
| \> |     \-H "X-Hume-Api-Key: \<apiKey\>" \\ |
| \> |     \-H "Content-Type: application/json" \\ |
| \> |     \-d '{ |
| \> |  "urls": \[ |
| \> |    "https://hume-tutorials.s3.amazonaws.com/faces.zip" |
| \> |  \], |
| \> |  "notify": true |
| \> | }' |

[Try it](https://dev.hume.ai/reference/expression-measurement-api/batch/start-inference-job?explorer=true)  
200Successful

| 1 | { |
| :---- | :---- |
| 2 |  "job\_id": "job\_id" |
| 3 | } |

Start a new measurement inference job.

### **Headers**

X-Hume-Api-KeystringRequired

### **Request**

This endpoint expects an object.

modelsobjectOptional

Specify the models to use for inference. If this field is not explicitly set, then all models will run by default.

Show 6 properties

transcriptionobjectOptional

Show 3 properties

urlslist of stringsOptional

URLs to the media files to be processed. Each must be a valid public URL to a media file (see recommended input filetypes) or an archive (.zip, .tar.gz, .tar.bz2, .tar.xz) of media files.

If you wish to supply more than 100 URLs, consider providing them as an archive (.zip, .tar.gz, .tar.bz2, .tar.xz).

textlist of stringsOptional

Text supplied directly to our Emotional Language and NER models for analysis.

callback\_urlstringOptionalformat: "url"

If provided, a POST request will be made to the URL with the generated predictions on completion or the error message on failure.

notifybooleanOptionalDefaults to false

Whether to send an email notification to the user upon job completion/failure.

### **Response**

job\_idstringformat: "uuid"

The ID of the started job.

Was this page helpful?

YesNo

[Previous](https://dev.hume.ai/reference/expression-measurement-api/batch/list-jobs)

#### [**Get job details**](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-details)

[Next](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-details)

[Built with](https://buildwithfern.com/?utm_campaign=buildWith&utm_medium=docs&utm_source=dev.hume.ai)

\*\*\*

**Get job details**  
**GET**  
https://api.hume.ai/v0/batch/jobs/:id  
**GET**  
/v0/batch/jobs/:id  
cURL

| $ | curl https://api.hume.ai \\ |
| :---- | :---- |
| \> |     \-H "X-Hume-Api-Key: \<apiKey\>" |

[Try it](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-details?explorer=true)  
200Retrieved

| 1 | { |
| :---- | :---- |
| 2 |  "type": "INFERENCE", |
| 3 |  "job\_id": "job\_id", |
| 4 |  "request": { |
| 5 |    "callback\_url": null, |
| 6 |    "files": \[\], |
| 7 |    "models": { |
| 8 |      "burst": {}, |
| 9 |      "face": { |
| 10 |        "descriptions": null, |
| 11 |        "facs": null, |
| 12 |        "fps\_pred": 3, |
| 13 |        "identify\_faces": false, |
| 14 |        "min\_face\_size": 60, |
| 15 |        "prob\_threshold": 0.99, |
| 16 |        "save\_faces": false |
| 17 |      }, |
| 18 |      "facemesh": {}, |
| 19 |      "language": { |
| 20 |        "granularity": "word", |
| 21 |        "identify\_speakers": false, |
| 22 |        "sentiment": null, |
| 23 |        "toxicity": null |
| 24 |      }, |
| 25 |      "ner": { |
| 26 |        "identify\_speakers": false |
| 27 |      }, |
| 28 |      "prosody": { |
| 29 |        "granularity": "utterance", |
| 30 |        "identify\_speakers": false, |
| 31 |        "window": null |
| 32 |      } |
| 33 |    }, |
| 34 |    "notify": true, |
| 35 |    "text": \[\], |
| 36 |    "urls": \[ |
| 37 |      "https://hume-tutorials.s3.amazonaws.com/faces.zip" |
| 38 |    \] |
| 39 |  }, |
| 40 |  "state": { |
| 41 |    "created\_timestamp\_ms": 1712590457884, |
| 42 |    "ended\_timestamp\_ms": 1712590462252, |
| 43 |    "num\_errors": 0, |
| 44 |    "num\_predictions": 10, |
| 45 |    "started\_timestamp\_ms": 1712590457995, |
| 46 |    "status": "COMPLETED" |
| 47 |  } |
| 48 | } |

Get the request details and state of a given job.

### **Path parameters**

idstringRequiredformat: "uuid"

The unique identifier for the job.

### **Headers**

X-Hume-Api-KeystringRequired

### **Response**

object

Show 4 properties

Was this page helpful?

YesNo

[Previous](https://dev.hume.ai/reference/expression-measurement-api/batch/start-inference-job)

#### [**Get job predictions**](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-predictions)

[Next](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-predictions)

[Built with](https://buildwithfern.com/?utm_campaign=buildWith&utm_medium=docs&utm_source=dev.hume.ai)

\*\*\*

[Expression Measurement APIBatch](https://dev.hume.ai/reference/expression-measurement-api/batch/list-jobs)  
**Get job predictions**  
**GET**  
https://api.hume.ai/v0/batch/jobs/:id/predictions  
**GET**  
/v0/batch/jobs/:id/predictions  
cURL

| $ | curl https://api.hume.ai \\ |
| :---- | :---- |
| \> |     \-H "X-Hume-Api-Key: \<apiKey\>" |

[Try it](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-predictions?explorer=true)  
200Retrieved

| 1 | \[ |
| :---- | :---- |
| 2 |  { |
| 3 |    "source": { |
| 4 |      "type": "url", |
| 5 |      "url": "https://hume-tutorials.s3.amazonaws.com/faces.zip" |
| 6 |    }, |
| 7 |    "results": { |
| 8 |      "predictions": \[ |
| 9 |        { |
| 10 |          "file": "faces/100.jpg", |
| 11 |          "models": { |
| 12 |            "face": { |
| 13 |              "metadata": null, |
| 14 |              "grouped\_predictions": \[ |
| 15 |                { |
| 16 |                  "id": "unknown", |
| 17 |                  "predictions": \[ |
| 18 |                    { |
| 19 |                      "frame": 0, |
| 20 |                      "time": 0, |
| 21 |                      "prob": 0.9994111061096191, |
| 22 |                      "box": { |
| 23 |                        "x": 1187.885986328125, |
| 24 |                        "y": 1397.697509765625, |
| 25 |                        "w": 1401.668701171875, |
| 26 |                        "h": 1961.424560546875 |
| 27 |                      }, |
| 28 |                      "emotions": \[ |
| 29 |                        { |
| 30 |                          "name": "Admiration", |
| 31 |                          "score": 0.10722749680280685 |
| 32 |                        }, |
| 33 |                        { |
| 34 |                          "name": "Adoration", |
| 35 |                          "score": 0.06395940482616425 |
| 36 |                        }, |
| 37 |                        { |
| 38 |                          "name": "Aesthetic Appreciation", |
| 39 |                          "score": 0.05811462551355362 |
| 40 |                        }, |
| 41 |                        { |
| 42 |                          "name": "Amusement", |
| 43 |                          "score": 0.14187128841876984 |
| 44 |                        }, |
| 45 |                        { |
| 46 |                          "name": "Anger", |
| 47 |                          "score": 0.02804684266448021 |
| 48 |                        }, |
| 49 |                        { |
| 50 |                          "name": "Anxiety", |
| 51 |                          "score": 0.2713485360145569 |
| 52 |                        }, |
| 53 |                        { |
| 54 |                          "name": "Awe", |
| 55 |                          "score": 0.33812594413757324 |
| 56 |                        }, |
| 57 |                        { |
| 58 |                          "name": "Awkwardness", |
| 59 |                          "score": 0.1745193600654602 |
| 60 |                        }, |
| 61 |                        { |
| 62 |                          "name": "Boredom", |
| 63 |                          "score": 0.23600080609321594 |
| 64 |                        }, |
| 65 |                        { |
| 66 |                          "name": "Calmness", |
| 67 |                          "score": 0.18988418579101562 |
| 68 |                        }, |
| 69 |                        { |
| 70 |                          "name": "Concentration", |
| 71 |                          "score": 0.44288986921310425 |
| 72 |                        }, |
| 73 |                        { |
| 74 |                          "name": "Confusion", |
| 75 |                          "score": 0.39346569776535034 |
| 76 |                        }, |
| 77 |                        { |
| 78 |                          "name": "Contemplation", |
| 79 |                          "score": 0.31002455949783325 |
| 80 |                        }, |
| 81 |                        { |
| 82 |                          "name": "Contempt", |
| 83 |                          "score": 0.048870109021663666 |
| 84 |                        }, |
| 85 |                        { |
| 86 |                          "name": "Contentment", |
| 87 |                          "score": 0.0579497292637825 |
| 88 |                        }, |
| 89 |                        { |
| 90 |                          "name": "Craving", |
| 91 |                          "score": 0.06544201076030731 |
| 92 |                        }, |
| 93 |                        { |
| 94 |                          "name": "Desire", |
| 95 |                          "score": 0.05526508390903473 |
| 96 |                        }, |
| 97 |                        { |
| 98 |                          "name": "Determination", |
| 99 |                          "score": 0.08590991795063019 |
| 100 |                        }, |
| 101 |                        { |
| 102 |                          "name": "Disappointment", |
| 103 |                          "score": 0.19508258998394012 |
| 104 |                        }, |
| 105 |                        { |
| 106 |                          "name": "Disgust", |
| 107 |                          "score": 0.031529419124126434 |
| 108 |                        }, |
| 109 |                        { |
| 110 |                          "name": "Distress", |
| 111 |                          "score": 0.23210826516151428 |
| 112 |                        }, |
| 113 |                        { |
| 114 |                          "name": "Doubt", |
| 115 |                          "score": 0.3284550905227661 |
| 116 |                        }, |
| 117 |                        { |
| 118 |                          "name": "Ecstasy", |
| 119 |                          "score": 0.040716782212257385 |
| 120 |                        }, |
| 121 |                        { |
| 122 |                          "name": "Embarrassment", |
| 123 |                          "score": 0.1467227339744568 |
| 124 |                        }, |
| 125 |                        { |
| 126 |                          "name": "Empathic Pain", |
| 127 |                          "score": 0.07633581757545471 |
| 128 |                        }, |
| 129 |                        { |
| 130 |                          "name": "Entrancement", |
| 131 |                          "score": 0.16245244443416595 |
| 132 |                        }, |
| 133 |                        { |
| 134 |                          "name": "Envy", |
| 135 |                          "score": 0.03267110139131546 |
| 136 |                        }, |
| 137 |                        { |
| 138 |                          "name": "Excitement", |
| 139 |                          "score": 0.10656816512346268 |
| 140 |                        }, |
| 141 |                        { |
| 142 |                          "name": "Fear", |
| 143 |                          "score": 0.3115977346897125 |
| 144 |                        }, |
| 145 |                        { |
| 146 |                          "name": "Guilt", |
| 147 |                          "score": 0.11615975946187973 |
| 148 |                        }, |
| 149 |                        { |
| 150 |                          "name": "Horror", |
| 151 |                          "score": 0.19795553386211395 |
| 152 |                        }, |
| 153 |                        { |
| 154 |                          "name": "Interest", |
| 155 |                          "score": 0.3136432468891144 |
| 156 |                        }, |
| 157 |                        { |
| 158 |                          "name": "Joy", |
| 159 |                          "score": 0.06285581737756729 |
| 160 |                        }, |
| 161 |                        { |
| 162 |                          "name": "Love", |
| 163 |                          "score": 0.06339752674102783 |
| 164 |                        }, |
| 165 |                        { |
| 166 |                          "name": "Nostalgia", |
| 167 |                          "score": 0.05866732448339462 |
| 168 |                        }, |
| 169 |                        { |
| 170 |                          "name": "Pain", |
| 171 |                          "score": 0.07684041559696198 |
| 172 |                        }, |
| 173 |                        { |
| 174 |                          "name": "Pride", |
| 175 |                          "score": 0.026822954416275024 |
| 176 |                        }, |
| 177 |                        { |
| 178 |                          "name": "Realization", |
| 179 |                          "score": 0.30000734329223633 |
| 180 |                        }, |
| 181 |                        { |
| 182 |                          "name": "Relief", |
| 183 |                          "score": 0.04414166510105133 |
| 184 |                        }, |
| 185 |                        { |
| 186 |                          "name": "Romance", |
| 187 |                          "score": 0.042728863656520844 |
| 188 |                        }, |
| 189 |                        { |
| 190 |                          "name": "Sadness", |
| 191 |                          "score": 0.14773206412792206 |
| 192 |                        }, |
| 193 |                        { |
| 194 |                          "name": "Satisfaction", |
| 195 |                          "score": 0.05902980640530586 |
| 196 |                        }, |
| 197 |                        { |
| 198 |                          "name": "Shame", |
| 199 |                          "score": 0.08103451132774353 |
| 200 |                        }, |
| 201 |                        { |
| 202 |                          "name": "Surprise (negative)", |
| 203 |                          "score": 0.25518184900283813 |
| 204 |                        }, |
| 205 |                        { |
| 206 |                          "name": "Surprise (positive)", |
| 207 |                          "score": 0.28845661878585815 |
| 208 |                        }, |
| 209 |                        { |
| 210 |                          "name": "Sympathy", |
| 211 |                          "score": 0.062488824129104614 |
| 212 |                        }, |
| 213 |                        { |
| 214 |                          "name": "Tiredness", |
| 215 |                          "score": 0.1559651643037796 |
| 216 |                        }, |
| 217 |                        { |
| 218 |                          "name": "Triumph", |
| 219 |                          "score": 0.01955239288508892 |
| 220 |                        } |
| 221 |                      \], |
| 222 |                      "facs": null, |
| 223 |                      "descriptions": null |
| 224 |                    } |
| 225 |                  \] |
| 226 |                } |
| 227 |              \] |
| 228 |            } |
| 229 |          } |
| 230 |        } |
| 231 |      \], |
| 232 |      "errors": \[\] |
| 233 |    } |
| 234 |  } |
| 235 | \] |

Get the JSON predictions of a completed inference job.

### **Path parameters**

idstringRequiredformat: "uuid"

The unique identifier for the job.

### **Headers**

X-Hume-Api-KeystringRequired

### **Response**

object

Hide 3 properties

sourceobject

Hide 3 variants

urlobject

Hide 2 properties

type"url"

urlstring

The URL of the source media file.

OR

fileobject

The list of files submitted for analysis.

Hide 4 properties

type"file"

md5sumstring

The MD5 checksum of the file.

content\_typestring or null

The content type of the file.

filenamestring or null

The name of the file.

OR

textobject

Hide 1 properties

type"text"

resultsobject or null

Hide 2 properties

predictionslist of objects

Hide 2 properties

filestring

A file path relative to the top level source URL or file.

modelsobject

Hide 6 properties

faceobject or null

Hide 2 properties

grouped\_predictionslist of objects

Hide 2 properties

idstring

An automatically generated label to identify individuals in your media file. Will be unknown if you have chosen to disable identification, or if the model is unable to distinguish between individuals.

predictionslist of objects

Hide 7 properties

frameuint64

Frame number

timedouble

Time in seconds when face detection occurred.

probdouble

The predicted probability that a detected face was actually a face.

boxobject

A bounding box around a face.

Show 4 properties

emotionslist of objects

A high-dimensional embedding in emotion space.

Show 2 properties

facslist of objects or null

FACS 2.0 features and their scores.

Show 2 properties

descriptionslist of objects or null

Modality-specific descriptive features and their scores.

Show 2 properties

metadataobject or null

No associated metadata for this model. Value will be null.

burstobject or null

Hide 2 properties

grouped\_predictionslist of objects

Hide 2 properties

idstring

An automatically generated label to identify individuals in your media file. Will be unknown if you have chosen to disable identification, or if the model is unable to distinguish between individuals.

predictionslist of objects

Hide 3 properties

timeobject

A time range with a beginning and end, measured in seconds.

Hide 2 properties

begindouble

Beginning of time range in seconds.

enddouble

End of time range in seconds.

emotionslist of objects

A high-dimensional embedding in emotion space.

Hide 2 properties

namestring

Name of the emotion being expressed.

scoredouble

Embedding value for the emotion being expressed.

descriptionslist of objects

Modality-specific descriptive features and their scores.

Hide 2 properties

namestring

Name of the descriptive feature being expressed.

scoredouble

Embedding value for the descriptive feature being expressed.

metadataobject or null

No associated metadata for this model. Value will be null.

prosodyobject or null

Hide 2 properties

grouped\_predictionslist of objects

Hide 2 properties

idstring

An automatically generated label to identify individuals in your media file. Will be unknown if you have chosen to disable identification, or if the model is unable to distinguish between individuals.

predictionslist of objects

Hide 5 properties

timeobject

A time range with a beginning and end, measured in seconds.

Show 2 properties

emotionslist of objects

A high-dimensional embedding in emotion space.

Show 2 properties

textstring or null

A segment of text (like a word or a sentence).

confidencedouble or null

Value between 0.0 and 1.0 that indicates our transcription model’s relative confidence in this text.

speaker\_confidencedouble or null

Value between 0.0 and 1.0 that indicates our transcription model’s relative confidence that this text was spoken by this speaker.

metadataobject or null

Transcription metadata for your media file.

Hide 2 properties

confidencedouble

Value between 0.0 and 1.0 indicating our transcription model’s relative confidence in the transcription of your media file.

detected\_languageobject or null

languageobject or null

Hide 2 properties

grouped\_predictionslist of objects

Show 2 properties

metadataobject or null

Transcription metadata for your media file.

Hide 2 properties

confidencedouble

Value between 0.0 and 1.0 indicating our transcription model’s relative confidence in the transcription of your media file.

detected\_languageobject or null

nerobject or null

Hide 2 properties

grouped\_predictionslist of objects

Hide 2 properties

idstring

An automatically generated label to identify individuals in your media file. Will be unknown if you have chosen to disable identification, or if the model is unable to distinguish between individuals.

predictionslist of objects

Hide 10 properties

entitystring

The recognized topic or entity.

positionobject

Position of a segment of text within a larger document, measured in characters. Uses zero-based indexing. The beginning index is inclusive and the end index is exclusive.

Show 2 properties

entity\_confidencedouble

Our NER model's relative confidence in the recognized topic or entity.

supportdouble

A measure of how often the entity is linked to by other entities.

uristring

A URL which provides more information about the recognized topic or entity.

link\_wordstring

The specific word to which the emotion predictions are linked.

emotionslist of objects

A high-dimensional embedding in emotion space.

Hide 2 properties

namestring

Name of the emotion being expressed.

scoredouble

Embedding value for the emotion being expressed.

timeobject or null

A time range with a beginning and end, measured in seconds.

Hide 2 properties

begindouble

Beginning of time range in seconds.

enddouble

End of time range in seconds.

confidencedouble or null

Value between 0.0 and 1.0 that indicates our transcription model’s relative confidence in this text.

speaker\_confidencedouble or null

Value between 0.0 and 1.0 that indicates our transcription model’s relative confidence that this text was spoken by this speaker.

metadataobject or null

Transcription metadata for your media file.

Hide 2 properties

confidencedouble

Value between 0.0 and 1.0 indicating our transcription model’s relative confidence in the transcription of your media file.

detected\_languageobject or null

facemeshobject or null

Hide 2 properties

grouped\_predictionslist of objects

Hide 2 properties

idstring

An automatically generated label to identify individuals in your media file. Will be unknown if you have chosen to disable identification, or if the model is unable to distinguish between individuals.

predictionslist of objects

Hide 1 properties

emotionslist of objects

A high-dimensional embedding in emotion space.

Hide 2 properties

namestring

Name of the emotion being expressed.

scoredouble

Embedding value for the emotion being expressed.

metadataobject or null

No associated metadata for this model. Value will be null.

errorslist of objects

Hide 2 properties

messagestring

An error message.

filestring

A file path relative to the top level source URL or file.

errorstring or null

An error message.

Was this page helpful?

YesNo

[Previous](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-details)

#### [**Get job artifacts**](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-artifacts)

[Next](https://dev.hume.ai/reference/expression-measurement-api/batch/get-job-artifacts)

[Built with](https://buildwithfern.com/?utm_campaign=buildWith&utm_medium=docs&utm_source=dev.hume.ai)

