# Industrie 4.0 - Modelle

Es wird für Heißkurven, die mit dem TCLAB aufgezeichnet wurden,  untersucht wie der Kurvenverlauf sich mittels neuronalen ODE (nODE) und physikalisch informierten Neuronalen Netzwerken (PINN) vorhersagen lassen.

Es wurden zwölf Heizyklen durchgeführt bei dem an der Wärmequelle Q1 zu 100 % aufgeheizt wird. Eine Heizphase beträgt etwa 300 Datenpunkte, danach wird für etwa 420 Datenpunkte das Heizelement sich abkühlen gelassen. Danach beginnt ein weiterer Messzyklus aus Aufheiz- und Abkühlungsphase.

Zwischen zwei Messpunkten liegen 1,1 bis 1,3 Sekunden. Diese Zeit setzt sich aus Messzeit, und Wartezeit von einer Sekunde zusammen. 

Der Temperatursensor hat einen typischen Messfehler von 1 K und einen maximalen Fehler von 3 K. Der höchste gemessene Temperaturanstieg war 1.6 Kelvin pro Sekunde, der höchste Abfall - 2.5 Kelvin pro Sekunde.



Für jeden Messpunkt werden notiert: 

- Temperatur am Sensor 1

- Tempertur am Sensor 2

- Prozentzahl  für Heizelement

- Systemzeit

- Heizzyklus

Die Pluto-Notebooks/ bzw. daraus erstellte pdf-Dokumente bauen sehr loose auf einander auf:

1) [nODE mit Flux](neuralODE/nODE_Flux—Pluto_Sebastian_Heiden.pdf)

2) [nODE mit Lux](neuralODE/nODE_Lux—Pluto_Sebastian_Heiden.pdf)

3) [PINN](PINN/PINN—Pluto_Sebastian_Heiden.pdf)