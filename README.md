# 🎙️ Alexa Voice Assistant Using Python and Machine Learning

## 📌 Project Overview

This project is a **voice-based AI assistant built using Python, Speech Recognition, Natural Language Processing (NLP), and Machine Learning**.

The assistant listens to the user's voice through a microphone, converts speech into text using **OpenAI Whisper**, identifies the user's intent using **TF-IDF and Multinomial Neural Network classification**, performs the requested action, and responds using **Text-to-Speech**.

The project is inspired by voice assistants such as Alexa and demonstrates how different AI technologies can be combined to create an interactive voice-controlled assistant.

---

## 🎯 Project Objective

The main objective of this project is to build a Python-based voice assistant capable of:

* Listening to user commands through a microphone.
* Converting voice input into text.
* Understanding the user's intention.
* Performing predefined actions based on the detected intent.
* Opening websites and search results.
* Playing music.
* Providing jokes and interesting facts.
* Giving date and time information.
* Simulating alarms, timers, reminders, and calendar actions.
* Searching for weather information.
* Providing directions using Google Maps.
* Managing a shopping list.
* Performing basic calculations.
* Responding to general questions.
* Converting text responses into speech.

---

## 🛠️ Technologies Used

The project uses the following technologies and Python libraries:

* **Python** – Main programming language
* **OpenAI Whisper** – Speech-to-text conversion
* **SoundDevice** – Audio recording from the microphone
* **SciPy** – Saving recorded audio as WAV files
* **TF-IDF Vectorizer** – Converting text commands into numerical features
* **MLPClassifier** – Machine learning model for intent classification
* **Pyttsx3** – Text-to-speech conversion
* **Pandas** – Dataset handling
* **NumPy** – Numerical operations
* **Webbrowser** – Opening websites and search results
* **PyAutoGUI** – Desktop automation support
* **Regular Expressions (re)** – Extracting information from user commands
* **SQLite/CSV Dataset** – Storing intent training examples
* **FFmpeg** – Audio processing support

---

## 🧠 System Architecture

The overall workflow of the assistant is:

```text
User Speaks
     ↓
Microphone
     ↓
Audio Recording
     ↓
Whisper Speech-to-Text
     ↓
Text Command
     ↓
TF-IDF Feature Extraction
     ↓
MLPClassifier
     ↓
Intent Detection
     ↓
Action Execution
     ↓
Response Generation
     ↓
Text-to-Speech
     ↓
Assistant Speaks
```

---

## 🔄 Project Workflow

### Step 1: Record User Voice

The assistant records audio from the microphone using the `sounddevice` library.

```python
def record_audio(filename="input.wav", duration=5, fs=48000):
    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1
    )
    sd.wait()
    write(filename, fs, recording)
```

The recorded audio is saved as a `.wav` file.

---

### Step 2: Speech-to-Text Conversion

The recorded audio is processed using the **OpenAI Whisper** model.

```python
model = whisper.load_model("base")
result = model.transcribe("input.wav")
```

Whisper converts the user's spoken command into text.

For example:

```text
User Voice:
"Set alarm for 7 AM"

Converted Text:
"set alarm for 7 AM"
```

The converted text is then passed to the intent classification model.

---

### Step 3: Intent Classification

The project uses a dataset containing example prompts and their corresponding intents.

Example:

| Prompt               | Intent       |
| -------------------- | ------------ |
| Play some music      | play_music   |
| Open YouTube         | open_website |
| Tell me a joke       | jokes_fun    |
| What is the weather? | weather      |
| Set alarm for 7 AM   | alarm        |
| What time is it?     | date_time    |

The prompts are converted into numerical features using **TF-IDF Vectorization**.

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(alexa_df["prompt"])
```

The extracted features are then used to train an **MLPClassifier**.

```python
intent_model = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    max_iter=500
)

intent_model.fit(X, alexa_df["intent"])
```

The trained model predicts the user's intent.

---

## 🤖 Intent Prediction

The `predict_intent()` function takes the user's text and predicts what the user wants.

```python
def predict_intent(text):
    X_test = vectorizer.transform([text])
    intent = intent_model.predict(X_test)[0]
    return intent
```

For example:

```text
Input:
"Play a song for me"

Predicted Intent:
play_music
```

The predicted intent is then passed to the action-handling function.

---

## ⚙️ Action Execution

The `perform_action()` function decides what action should be performed based on the detected intent.

The assistant supports multiple commands.

### 🎵 Play Music

The assistant opens YouTube music search results.

```text
User:
"Play music"

Action:
Open YouTube music search
```

---

### 🌐 Open Websites

The assistant can open websites such as:

* YouTube
* Google
* GitHub

Example:

```text
User:
"Open YouTube"

Response:
"Opening YouTube"
```

---

### 😂 Jokes

The assistant can provide random programming-related jokes.

Example:

```text
User:
"Tell me a joke"

Response:
"Why do programmers prefer dark mode?
Because light attracts bugs!"
```

---

### 📰 News

The assistant opens Google News to display current headlines.

---

### 🎬 Movies

The assistant opens IMDb's popular movie chart.

---

### ⏰ Timer

The assistant extracts the number of minutes from the command and simulates setting a timer.

Example:

```text
User:
"Set a timer for 10 minutes"

Response:
"Timer set for 10 minutes (simulation)"
```

---

### ⏰ Alarm

The assistant identifies the requested alarm time using regular expressions.

Example:

```text
User:
"Set alarm for 7 AM"

Response:
"Alarm set for 7:00 AM (simulation)"
```

The current implementation simulates the alarm rather than creating a real operating system alarm.

---

### 🔔 Reminder

The assistant provides a simulated reminder response.

---

### 🕒 Date and Time

The assistant uses Python's `datetime` module to provide the current date and time.

Example:

```text
User:
"What time is it?"

Response:
"It's 10:30 AM on July 23, 2026"
```

---

### 📅 Calendar

The current implementation provides a simulated calendar response.

---

### 🌦️ Weather

The assistant identifies the requested city and opens a Google weather search.

Example:

```text
User:
"What is the weather in Delhi?"

Action:
Open Google weather search for Delhi
```

---

### 🧠 General Questions

For general questions, the assistant creates a Google search query and opens the results in the browser.

Example:

```text
User:
"Who is the Prime Minister of India?"

Action:
Search the question on Google
```

---

### 🌍 Facts

The assistant provides random interesting facts.

Example:

```text
"Honey never spoils."
```

---

### 🚗 Traffic

The assistant opens Google Maps for a predefined route.

---

### 🧭 Directions

The assistant extracts a destination from the user's command and opens Google Maps.

Example:

```text
User:
"Directions to Bhubaneswar"

Action:
Open Google Maps search
```

---

### 🛒 Shopping List

The assistant maintains a temporary shopping list during the program execution.

Example:

```text
User:
"Add milk to my shopping list"

Response:
"Added milk to your shopping list"
```

---

### 🏠 Smart Home

The project includes a simulated smart-home command handler that can be extended in the future to control IoT devices.

---

### 🤖 Personality

The assistant can respond to questions about itself using predefined responses.

Example:

```text
User:
"How are you?"

Response:
"I'm doing great! Ready to help you."
```

---

### 🧮 Calculator

The assistant supports basic arithmetic operations such as:

* Addition
* Subtraction
* Multiplication
* Division

Example:

```text
User:
"What is 10 plus 20?"

Response:
"The answer is 30"
```

---

## 🔊 Text-to-Speech

After performing an action, the assistant converts the response text into speech using **pyttsx3**.

```python
engine = pyttsx3.init("sapi5")
```

The `speak()` function handles voice output:

```python
def speak(text):
    print("Alexa:", text)
    engine.say(text)
    engine.runAndWait()
```

This allows the assistant to communicate with the user through voice.

---

## 🚀 Main Assistant Pipeline

The complete assistant is executed using:

```python
def run_assistant():
    record_audio("input.wav")
    text = speech_to_text("input.wav")
    intent = predict_intent(text)
    response = perform_action(intent, text)
    speak(response)
```

The process is:

```text
Record Audio
     ↓
Speech-to-Text
     ↓
Intent Prediction
     ↓
Perform Action
     ↓
Generate Response
     ↓
Text-to-Speech
```

---

## 📁 Project Structure

A possible project structure is:

```text
Alexa Voice Assistant/
│
├── input.wav
├── alexa_data.csv
├── assistant.py
├── ffmpeg/
│   └── ffmpeg-build/
│
└── README.md
```

---

## 📊 Machine Learning Model

The project uses an **MLPClassifier (Multi-Layer Perceptron Classifier)** for intent classification.

The model architecture contains:

```text
Input Layer
     ↓
Hidden Layer: 50 Neurons
     ↓
Hidden Layer: 25 Neurons
     ↓
Output Layer
```

The output represents the predicted intent of the user's command.

---

## 💡 Key Features

* 🎤 Voice input through microphone
* 📝 Speech-to-text conversion
* 🧠 Machine learning-based intent classification
* 🔤 TF-IDF text feature extraction
* 🤖 MLP neural network classifier
* 🔊 Text-to-speech responses
* 🎵 Music search
* 🌐 Website opening
* 📰 News search
* 🌦️ Weather search
* 🧭 Google Maps directions
* ⏰ Alarm and timer simulation
* 🔔 Reminder simulation
* 🛒 Shopping list
* 🧮 Basic calculator
* 😂 Jokes and facts
* 🕒 Date and time
* 🔎 General web search

---

## 🔧 Installation

Install the required Python libraries:

```bash
pip install pandas numpy scipy sounddevice openai-whisper pyttsx3 scikit-learn pyautogui
```

FFmpeg may also be required for audio processing.

After installing FFmpeg, add its `bin` directory to the system `PATH`.

---

## ▶️ How to Run

1. Install Python.
2. Install the required dependencies.
3. Install and configure FFmpeg.
4. Connect a working microphone.
5. Make sure `alexa_data.csv` is available.
6. Train the intent classification model.
7. Run the Python script.
8. Speak a command when the assistant displays:

```text
🎤 Listening...
```

9. The assistant will process the command and provide a voice response.

---

## 📝 Example Commands

You can try commands such as:

```text
"Play music"
"Open YouTube"
"Open Google"
"Tell me a joke"
"What is the weather in Delhi?"
"Set alarm for 7 AM"
"Set a timer for 10 minutes"
"What time is it?"
"Who is the Prime Minister of India?"
"Give me a fact"
"Directions to Bhubaneswar"
"Add milk to my shopping list"
"What is 10 plus 20?"
"How are you?"
```

---

## ⚠️ Current Limitations

Some features in the current version are simulations rather than fully integrated services:

* Alarm is simulated.
* Timer is simulated.
* Reminder is simulated.
* Calendar is simulated.
* Smart home control is simulated.
* Weather information is retrieved through a web search rather than a dedicated weather API.
* General questions are handled through Google search.
* Traffic uses a predefined route.
* The shopping list is stored only in memory and is lost when the program stops.

---

## 🚀 Future Improvements

The project can be improved by adding:

* Continuous voice listening.
* Wake-word detection such as "Alexa".
* Real-time alarm functionality.
* Real-time timers and reminders.
* Google Calendar integration.
* Weather API integration.
* News API integration.
* Spotify or YouTube Music integration.
* Persistent shopping list storage.
* Smart home IoT integration.
* Better intent classification using transformer models.
* Large Language Model integration for natural conversations.
* User authentication and personalized responses.
* GUI or web-based interface.
* Deployment as a desktop or mobile application.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Python programming
* Speech recognition
* Audio recording
* Speech-to-text processing
* Natural Language Processing
* TF-IDF feature extraction
* Machine Learning classification
* Neural Networks
* Intent classification
* Text-to-speech conversion
* Regular expressions
* Browser automation
* Desktop automation
* Building an end-to-end AI assistant

---

## 📌 Project Summary

This project demonstrates how **Speech Recognition, NLP, Machine Learning, and Text-to-Speech technologies** can be combined to create a voice-controlled AI assistant.

The assistant takes voice input from the user, converts it into text using **OpenAI Whisper**, predicts the user's intent using **TF-IDF and an MLPClassifier**, performs the appropriate action, and finally responds using **pyttsx3**.

The project provides practical experience in building an end-to-end AI application and can be further extended into a more advanced personal voice assistant with real-time APIs, LLM integration, smart home control, and persistent user preferences.

---

## 📝 Resume Description

> Developed a Python-based AI Voice Assistant using OpenAI Whisper for speech-to-text conversion, TF-IDF for text feature extraction, and MLPClassifier for intent classification. Implemented voice-based commands for web search, music, weather, directions, calculator, shopping list, jokes, and other assistant functionalities, with pyttsx3 for text-to-speech responses.

---

## ⭐ Short Project Description

> **AI Voice Assistant** is a Python-based voice assistant that converts speech into text using OpenAI Whisper, identifies user intent using TF-IDF and an MLP neural network, performs predefined actions, and responds using text-to-speech. The project demonstrates the integration of Speech Recognition, NLP, Machine Learning, and automation.

