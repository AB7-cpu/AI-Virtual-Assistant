import speech_recognition as sr
import pywhatkit
import os
import subprocess as sb
import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

r = sr.Recognizer()

def takeCommand():
    try:
        with sr.Microphone() as source:
            print('Listening...')
            audio = r.listen(source)
            print('Recognizing...')
            query = r.recognize_google(audio)
            print(f'user said: {query}')
            return query.lower()
    
    except:
        print('Say that again please...')

if __name__ == '__main__':
    while True:
        query = takeCommand()

        if query!=None:

            if 'play' in query:
                query = query.replace('play', '')
                print(f'Assistant: Playing {query} on youtube')
                speak(f'Playing {query} on youtube')
                pywhatkit.playonyt(query)
                
            elif 'open' in query:
                query = query.replace('open', '')

                if 'brave' in query:
                    print('Assistant: Opening Brave Browser.')
                    speak('Opening Brave Browser')
                    sb.Popen('C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe')

                elif 'zen' in query:
                    print('Assistant: Opening Zen Browser.')
                    speak('Opening Zen Browser')
                    sb.Popen('C:\\Program Files\\Zen Browser\\zen.exe')
            
            elif 'quit' in query:
                print('Assistant: Aborting system.')
                speak('Aborting system')
                os.abort()