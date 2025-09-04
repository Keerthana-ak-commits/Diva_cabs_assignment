# Diva Cabs Trip Optimizer

This project analyzes cab operator *trip offers* and *trip seeks* from a Google Sheet to find optimal **multi-leg chains** that benefit both operators and the platform.

---

## Objective

Identify chains of 3–4 (or more) cab trips that:
- Connect multiple operators' SEEKING and OFFERING data
- Form a **profitable full-day round-trip** for a single cab (no empty legs)
- Match time feasibility (drive time + buffer) between legs

---

## 🔍 What This Project Does

- Parses and cleans data from Excel (`Diva Cabs.xlsx`)
- Matches SEEK and OFFER trips between Tata and Ranchi
- Builds 3–4 leg **trip chains** based on time/location feasibility
- Estimates drive time, buffers, and calculates rough profitability
- # Outputs:
  - A validated chain CSV
  - Cleaned operator contact CSV
  - Full script to re-run or adapt

---

## Project Files

| File | Description |
|------|-------------|
| `diva_cabs_script.py` | Full Python logic to clean, match, and generate trip chains |
| `diva_best_chain.csv` | Final selected 4-leg chain that meets all constraints |
| `diva_best_chain_contacts.csv` | Clean contact list of matched operators |
| `README.md` 

---

## Assumptions Used

-  **Tata ↔ Ranchi drive time:** ~210 minutes
- **Buffer between legs:** 30 minutes minimum
- **Base Fare:** 35/km
- **Revenue rate:** ₹22/km
- **Cost rate:** ₹18/km
- **Distance per leg:** ~130 km

> These can be easily updated in the script.

---

## Estimated Profit for This Chain

- Legs: 4
- Total Distance: ~520 km
- No empty legs
- Revenue: ₹11580 
- Cost: ₹9360
- **Net Profit:** ₹2220

---

## Submission Links
- GitHub Repository: [(https://github.com/Keerthana-ak-commits/Diva_cabs_assignment/tree/main)]

---

## Credits

This project was created as part of an operational optimization challenge by **DiggiPlus**.

Crafted with and Python by Keerthana.
