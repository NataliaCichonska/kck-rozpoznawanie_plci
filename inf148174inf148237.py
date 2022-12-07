
import numpy as np
import os
import sys
import soundfile as sf
from scipy.fftpack import fft, fftfreq
from scipy.signal import decimate

Mmin=85
Mmax=171
Kmin=171
Kmax=255

def hps(fft, ind, jnd):
    fft_kopia = fft.copy()
    for i in range(2,5):
        d=decimate(fft,i)
        #domnażanie widma decymowanych wersji do modułu oryginalnego
        for j in range(0,len(d)):
            fft_kopia[j] *=d[j]
    #tłumienie szumów i wybór podstawowej częstotliwości
    fft_kopia[:ind] = 0
    fft_kopia[jnd:] = 0
    indeks = np.argmax(fft_kopia)
    return indeks

def uzyskaj_indeks_czestotliwosci_podstawowej(sygnal, i, j):
    #kaiser
    sygnal=sygnal*np.kaiser(len(sygnal),beta=14)
    sygnal=abs(fft(sygnal))
    #hps
    indeks=hps(sygnal, i, j)
    return indeks

#zwraca sygnał i częstotliwość próbkowania
def czytaj_wav(nazwa_pliku):
    sygnal, czestotliowsc_prob = sf.read("train/"+nazwa_pliku)
    if(sygnal.ndim==2):
        #konwersja do sygnału mono
        sygnal=(sygnal[:,0] + sygnal[:,1])/2
    return sygnal, czestotliowsc_prob

#zwraca "M", jeśli głos należy do mężczyzny, a "K", jeśli należy do kobiety
def rozpoznaj(nazwa_pliku):
    sygnal, czestotliwosc_prob = czytaj_wav(nazwa_pliku)
    fx = np.linspace(0,czestotliwosc_prob,num=len(sygnal),endpoint=False)

    #wyznaczanie indeksów szumu
    for i in range(len(fx)):
        if (fx[i]>Mmin):
            break
    for j in range(i,len(fx)):
        if (fx[j]>Kmax):
            break

    indeks =uzyskaj_indeks_czestotliwosci_podstawowej(sygnal, i,j)
    czestotliowsc_podstawowa= fx[indeks]

    if (czestotliowsc_podstawowa<=Mmax):
        return "M"
    else :
        return "K"

if __name__ == '__main__':
        nazwa_pliku = sys.argv[1]
        if(nazwa_pliku=="all"):
            wszystkie=0
            niepoprawne=0
            for filename in os.listdir('train'):
                wszystkie+=1
                wynik=rozpoznaj(filename)
                if(filename.split("_")[1].split(".")[0]!=wynik):
                    niepoprawne+=1
                    print(filename, wynik)
            print(niepoprawne,wszystkie)
            print("accuracy: ", (wszystkie-niepoprawne)/wszystkie)

        else:
            wynik = rozpoznaj(nazwa_pliku)
            print(wynik)