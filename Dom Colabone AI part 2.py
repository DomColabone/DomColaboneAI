import pyttsx3
import speech_recognition as sr
import datetime  # O datetime pode ser usado para mostrar a hora
# import os  # Removido porque não é necessário

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"Você disse: {command}")
            return command.lower()
        except sr.UnknownValueError:
            speak("Desculpe, não consegui entender o que você disse.")
            return None
        except sr.RequestError:
            speak("Não consegui me conectar ao serviço de reconhecimento de fala.")
            return None

def execute_command(command):
    if 'hora' in command:
        now = datetime.datetime.now()
        speak(f"Agora são {now.hour} horas e {now.minute} minutos.")
    elif 'abrir google' in command:
        webbrowser.open('https://www.google.com')
        speak("Abrindo o Google.")
    elif 'abrir notepad' in command:
        os.system('notepad')
        speak("Abrindo o Bloco de Notas.")
    elif 'sair' in command:
        speak("Encerrando. Até mais!")
        exit()
    else:
        speak("Desculpe, não entendi o comando.")

def main():
    speak("Olá! Como posso ajudá-lo hoje?")
    while True:
        command = listen()
        if command:
            execute_command(command)

if __name__ == "__main__":
    main()

audio = recognizer.listen(source, timeout=5)  # tempo limite de 5 segundos