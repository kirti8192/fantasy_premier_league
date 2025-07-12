# FPL Machine Learning Project

This project uses historical Fantasy Premier League (FPL) data to build predictive models that help with player selection decisions. Data sources include the official FPL API and the community-maintained [Vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) dataset.

## 🔍 Classification Problems to Explore

### 1. Binary Classification: "Buy or Not"
- **Goal:** Predict whether a player will score more than 4 points in the next gameweek.
- **Target:** `1` if `total_points_next_gw > 4`, else `0`
- **Use for:** Early decision-making on player selection
- **Model type:** Binary classifier

---

### 2. Multi-class Classification: "Points Tier"
- **Goal:** Predict the point bracket a player will fall into in the next gameweek.
- **Classes:**
  - `0`: 0–1 points
  - `1`: 2–4 points
  - `2`: 5–7 points
  - `3`: 8+ points
- **Use for:** Fine-grained performance prediction
- **Model type:** Multi-class classifier (4 classes)

---

### 3. Position Prediction: "Guess the Player’s Role"
- **Goal:** Predict a player's FPL position (GK, DEF, MID, FWD) from their stats.
- **Target:** `position` column (1–4)
- **Use for:** Feature exploration — does the model learn player roles from patterns?
- **Model type:** Multi-class classifier (4 classes)

---

### 4. Availability Prediction: "Will They Miss the Next GW?"
- **Goal:** Predict if a player will be unavailable next gameweek.
- **Target:** `1` if `minutes_next_gw == 0`, else `0`
- **Use for:** Simulating injury/rest/suspension likelihood
- **Model type:** Binary classifier

---

### 5. Form Classification: "Hot or Cold?"
- **Goal:** Predict if a player is in low, medium, or high form.
- **Form Metric:** Rolling average of `total_points` over past 3 gameweeks
- **Classes:**
  - `0`: form < 2
  - `1`: 2 ≤ form ≤ 5
  - `2`: form > 5
- **Use for:** Detecting trends and hot streaks
- **Model type:** Multi-class classifier (3 classes)

---

## 📂 Directory Structure (WIP)