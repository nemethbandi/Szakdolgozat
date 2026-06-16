# 📊 A volatilitás-előrejelzés gazdasági értéke

## Standard volatilitásmodellek összehasonlítása dinamikus portfólióallokációban

---

## 📑 Tartalomjegyzék

* [Áttekintés](#-áttekintés)
* [Kutatási cél](#-kutatási-cél)
* [Vizsgált modellek](#-vizsgált-modellek)
* [Módszertan](#-módszertan)
* [Volatilitás-előrejelzés értékelése](#-volatilitás-előrejelzés-értékelése)
* [Volatility Targeting stratégia](#-volatility-targeting-stratégia)
* [Portfólióteljesítmény értékelése](#-portfólióteljesítmény-értékelése)
* [Kutatási kérdések](#-kutatási-kérdések)
* [Felhasznált technológiák](#-felhasznált-technológiák)
* [Összefoglalás](#-összefoglalás)

---

## 🧠 Áttekintés

A szakdolgozat célja különböző standard volatilitás-előrejelző modellek gyakorlati alkalmazhatóságának vizsgálata pénzügyi piacokon. A kutatás középpontjában nem új modellek fejlesztése áll, hanem annak elemzése, hogy a meglévő módszerek milyen előrejelzési képességgel rendelkeznek, illetve ezek az előrejelzések milyen befektetési eredményekhez vezetnek.

---

## 🎯 Kutatási cél

A dolgozat célja:

* különböző volatilitás-előrejelző modellek összehasonlítása egynapos előrejelzési horizonton,
* a modellek előrejelzési képességének statisztikai vizsgálata,
* annak elemzése, hogy az előrejelzett volatilitás hogyan használható fel dinamikus portfólióallokáció során,
* annak meghatározása, hogy a pontosabb előrejelzés jelent-e gazdasági előnyt.

---

## ⚙️ Vizsgált modellek

### 📉 Klasszikus volatilitásmodellek

* GARCH(1,1)
* EGARCH(1,1)
* GJR-GARCH(1,1)

### 📐 Realized volatility modell

* HAR-RV (Heterogeneous Autoregressive Model)

### 🤖 Mélytanulási modellek

* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)

A modellek célja minden esetben a következő napi volatilitás előrejelzése.

---

## 🔁 Módszertan

A vizsgálat rolling, out-of-sample keretrendszerben történik.

A folyamat lépései:

1. A modellek becslése múltbeli adatok alapján.
2. Egynapos volatilitás-előrejelzés készítése.
3. Az előrejelzések statisztikai összehasonlítása.
4. Az előrejelzések alkalmazása egy volatility targeting stratégiában.
5. A portfóliók historikus visszatesztelése.
6. A stratégiák teljesítményének összehasonlítása.

---

## 📊 Volatilitás-előrejelzés értékelése

A modellek előrejelzési teljesítményének összehasonlítása különböző statisztikai mutatók segítségével történik, például:

* MAPE
* RMSE
* Patton-féle QLIKE veszteségfüggvény
* Mincer–Zarnowitz regresszió

Az első vizsgálati szakasz célja annak meghatározása, hogy a különböző modellek közül melyik rendelkezik jobb volatilitás-előrejelzési képességgel.

---

## 💼 Volatility Targeting stratégia

A becsült volatilitás alapján dinamikusan kerül meghatározásra a portfólió kockázati kitettsége annak érdekében, hogy a stratégia egy előre meghatározott célvolatilitást tartson fenn.

A stratégia célja:

* a kockázat kontrollálása,
* a túlzott volatilitás mérséklése,
* a kedvezőbb kockázat–hozam profil elérése.

---

## 📈 Portfólióteljesítmény értékelése

A volatility targeting stratégia eredményei az alábbi mutatók alapján kerülnek összehasonlításra:

* Teljes hozam
* Sharpe-ráta
* Sortino-ráta
* Maximum Drawdown

A második vizsgálati szakasz célja annak meghatározása, hogy a jobb előrejelzési teljesítmény valóban jobb befektetési eredményekhez vezet-e.

---

## 🔍 Kutatási kérdések

* Melyik modell képes a legpontosabb egynapos volatilitás-előrejelzésre?
* A mélytanulási modellek felülmúlják-e a hagyományos GARCH-modelleket?
* A jobb előrejelzési teljesítmény jobb portfólióteljesítményt eredményez-e?
* Mennyire használhatók a volatilitásmodellek gyakorlati befektetési döntéstámogatásra?

---

## 🛠️ Felhasznált technológiák

* Python
* PyTorch
* NumPy
* Pandas
* arch
* Matplotlib
* saját backtesting framework

---

## 📌 Összefoglalás

A dolgozat nem új volatilitásmodellek fejlesztésére törekszik, hanem azt vizsgálja, hogy a jelenleg elérhető klasszikus és mélytanulási módszerek milyen előrejelzési teljesítményt nyújtanak, illetve ezek az előrejelzések milyen eredményekhez vezetnek egy gyakorlati portfóliókezelési alkalmazásban. A kutatás két fő kérdésre keresi a választ: melyik modell becsli pontosabban a következő napi volatilitást, és ez a különbség megjelenik-e a befektetési teljesítményben is.

---

## 📬 Kapcsolat

Készítette: *Németh András Fülöp*
