## 📑 Tartalomjegyzék

- #áttekintés
- #cél
- #modellek
- #módszertan
- #portfólióallokáció
- #értékelési-metrikák
- #fő-kérdés
- #hozzájárulás
- #technológia
- #összefoglalás

# 📊 A volatilitás-előrejelzés gazdasági értéke  
## GARCH és mélytanulási modellek összehasonlítása dinamikus portfólióallokációban

---

## 🧠 Áttekintés

Ez a projekt a volatilitás-előrejelzés gyakorlati hasznosságát vizsgálja pénzügyi piacokon.  
A fókusz nem csupán az előrejelzési pontosságon van, hanem azon is, hogy ezek az előrejelzések milyen hatással vannak a befektetési döntésekre és a portfólió teljesítményére.

---

## 🎯 Cél

A kutatás célja:

- különböző volatilitás-előrejelző modellek összehasonlítása **egynapos horizonton**
- annak vizsgálata, hogy az eltérő előrejelzések hogyan befolyásolják a **dinamikus portfólióallokációt**
- annak megértése, hogy a jobb előrejelzés jelent-e **valós gazdasági előnyt**

---

## ⚙️ Modellek

### 📉 Klasszikus ökonometriai modellek
- GARCH  
- EGARCH  
- GJR-GARCH  

### 🤖 Mélytanulási modellek (PyTorch)
- LSTM (Long Short-Term Memory)  
- GRU (Gated Recurrent Unit)  

---

## 🔁 Módszertan

A modellek egy **rolling, out-of-sample** keretrendszerben kerülnek kiértékelésre:

- minden időpontban csak múltbeli adatok használata  
- **1 napos előrejelzési horizont**  
- folyamatos újrabecslés  

Az előrejelzett volatilitás alapján történik a portfólió súlyozása egy előre definiált:

> 🎯 **target volatilitás** elérése érdekében

---

## 💼 Portfólióallokáció

A modellek által becsült volatilitás alapján egy dinamikus stratégia határozza meg:

- az eszközbe fektetett tőkearányt  
- a kockázatvállalás mértékét  

---

## 📈 Értékelési metrikák

A backtestelt stratégiák teljesítménye az alábbi mutatók alapján kerül összehasonlításra:

- Sharpe-ráta  
- Sortino-ráta  
- Maximum Drawdown  
- Teljes hozam  

---

## 🔍 Fő kérdés

Képesek-e a mélytanulási modellek:

> **gazdasági értelemben felülmúlni a hagyományos volatilitásmodelleket?**

Azaz:

- jobb előrejelzés → jobb portfólió?
- vagy a klasszikus modellek továbbra is versenyképesek?

---

## 🧩 Hozzájárulás

A dolgozat fő újdonsága, hogy:

- nem csak statisztikai pontosságot vizsgál  
- hanem a modellek **valós befektetési teljesítményét** is elemzi  
- historikus adatokon végzett backtesztekkel  

---

## 🛠️ Technológia

- Python  
- PyTorch  
- NumPy / Pandas  
- saját backtesting framework  

---

## 📌 Összefoglalás

A projekt célja annak feltárása, hogy a modern AI-alapú modellek:

- ✔️ pontosabb előrejelzést adnak-e  
- ✔️ és ez valóban jobb befektetési döntésekhez vezet-e  

---

## 📬 Kapcsolat

Készítette: *Németh András Fülöp*
