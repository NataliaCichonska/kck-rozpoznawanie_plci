
import numpy as np
import os
import sys
import soundfile as sf
from scipy.fftpack import fft, fftfreq
from scipy.signal import decimate

Mmin=85
Mmax=176
Kmin=176
Kmax=255

def hps(fft):
    fft_kopia = fft.copy()
    for i in range(2,5):
        d=decimate(fft,i)
        #dodawaj widma decymowanych wersji do modułu oryginalnego
        for j in range(0,len(d)):
            fft_kopia[j] +=d[j]
    indeks = np.argmax(fft_kopia)
    return indeks

def uzyskaj_indeks_czestotliwosci_podstawowej(sygnal):
    #kaiser
    sygnal=sygnal*np.kaiser(len(sygnal),beta=14)
    sygnal=abs(fft(sygnal))
    #hps
    indeks=hps(sygnal)
    return indeks

#zwraca sygnał i częstotliwość próbkowania
def czytaj_wav(nazwa_pliku):
    sygnal, czestotliowsc_prob = sf.read("train/"+nazwa_pliku)
    if(sygnal.ndim==2):
        #konwersja do sygnału mono
        sygnal=(sygnal[:,0] + sygnal[:,1])/2
    return sygnal, czestotliowsc_prob

#printuje "M", jeśli głos należy do mężczyzny, a "K", jeśli należy do kobiety
def rozpoznaj(nazwa_pliku):
    sygnal, czestotliwosc_prob = czytaj_wav(nazwa_pliku)
    fx=fftfreq(len(sygnal), d=1/czestotliwosc_prob)
    indeks=uzyskaj_indeks_czestotliwosci_podstawowej(sygnal)
    czestotliowsc_podstawowa=fx[indeks]
    if (czestotliowsc_podstawowa>=Mmin and czestotliowsc_podstawowa<=Mmax):
        print("M")
        return("M")
    else:
        print("K")
        return("K")

if __name__ == '__main__':
        nazwa_pliku = sys.argv[1]
        if(nazwa_pliku=="all"):
            wszystkie=0
            niepoprawne=0
            for filename in os.listdir('train'):
                wszystkie+=1
                wynik=rozpoznaj(filename)
                if(filename.split("_")[1].split(".")[0]!=wynik):
                    # print("niepoprawne: "+filename)
                    niepoprawne+=1
            print(niepoprawne,wszystkie)

        else:
            rozpoznaj(nazwa_pliku)
