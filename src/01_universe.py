"""
Universe construction - S&P 500 Industrials sector (GICS 20).
Hardcoded snapshot for reproducibility. Composition is the constituent set
as of late 2024 / early 2025. Hardcoding is preferred for reproducibility:
the index drifts continuously, but the study must be replicable.
"""
import pandas as pd

INDUSTRIALS = [
    # Aerospace & Defense
    ("BA",   "Boeing",                       "Aerospace & Defense"),
    ("LMT",  "Lockheed Martin",              "Aerospace & Defense"),
    ("RTX",  "RTX Corp",                     "Aerospace & Defense"),
    ("NOC",  "Northrop Grumman",             "Aerospace & Defense"),
    ("GD",   "General Dynamics",             "Aerospace & Defense"),
    ("LHX",  "L3Harris Technologies",        "Aerospace & Defense"),
    ("HII",  "Huntington Ingalls",           "Aerospace & Defense"),
    ("TXT",  "Textron",                      "Aerospace & Defense"),
    ("TDG",  "TransDigm Group",              "Aerospace & Defense"),
    ("HWM",  "Howmet Aerospace",             "Aerospace & Defense"),
    # Air Freight & Logistics
    ("UPS",  "United Parcel Service",        "Air Freight & Logistics"),
    ("FDX",  "FedEx",                        "Air Freight & Logistics"),
    ("EXPD", "Expeditors International",     "Air Freight & Logistics"),
    ("CHRW", "C.H. Robinson",                "Air Freight & Logistics"),
    # Passenger Airlines
    ("DAL",  "Delta Air Lines",              "Passenger Airlines"),
    ("UAL",  "United Airlines",              "Passenger Airlines"),
    ("LUV",  "Southwest Airlines",           "Passenger Airlines"),
    ("AAL",  "American Airlines",            "Passenger Airlines"),
    ("ALK",  "Alaska Air Group",             "Passenger Airlines"),
    # Building Products
    ("CARR", "Carrier Global",               "Building Products"),
    ("OTIS", "Otis Worldwide",               "Building Products"),
    ("JCI",  "Johnson Controls",             "Building Products"),
    ("MAS",  "Masco",                        "Building Products"),
    ("AOS",  "A.O. Smith",                   "Building Products"),
    ("ALLE", "Allegion",                     "Building Products"),
    # Construction & Engineering
    ("PWR",  "Quanta Services",              "Construction & Engineering"),
    ("J",    "Jacobs Solutions",             "Construction & Engineering"),
    # Electrical Equipment
    ("ETN",  "Eaton",                        "Electrical Equipment"),
    ("EMR",  "Emerson Electric",             "Electrical Equipment"),
    ("ROK",  "Rockwell Automation",          "Electrical Equipment"),
    ("AME",  "Ametek",                       "Electrical Equipment"),
    ("HUBB", "Hubbell",                      "Electrical Equipment"),
    ("GNRC", "Generac Holdings",             "Electrical Equipment"),
    # Industrial Conglomerates
    ("GE",   "GE Aerospace",                 "Industrial Conglomerates"),
    ("HON",  "Honeywell",                    "Industrial Conglomerates"),
    ("MMM",  "3M",                           "Industrial Conglomerates"),
    ("ROP",  "Roper Technologies",           "Industrial Conglomerates"),
    # Machinery
    ("CAT",  "Caterpillar",                  "Machinery"),
    ("DE",   "Deere & Co",                   "Machinery"),
    ("ITW",  "Illinois Tool Works",          "Machinery"),
    ("PH",   "Parker-Hannifin",              "Machinery"),
    ("CMI",  "Cummins",                      "Machinery"),
    ("PCAR", "PACCAR",                       "Machinery"),
    ("DOV",  "Dover",                        "Machinery"),
    ("XYL",  "Xylem",                        "Machinery"),
    ("IR",   "Ingersoll Rand",               "Machinery"),
    ("FTV",  "Fortive",                      "Machinery"),
    ("GGG",  "Graco",                        "Machinery"),
    ("NDSN", "Nordson",                      "Machinery"),
    ("SNA",  "Snap-on",                      "Machinery"),
    ("PNR",  "Pentair",                      "Machinery"),
    ("SWK",  "Stanley Black & Decker",       "Machinery"),
    # Commercial Services
    ("WM",   "Waste Management",             "Commercial Services"),
    ("RSG",  "Republic Services",            "Commercial Services"),
    ("CTAS", "Cintas",                       "Commercial Services"),
    ("VRSK", "Verisk Analytics",             "Commercial Services"),
    ("CPRT", "Copart",                       "Commercial Services"),
    ("RHI",  "Robert Half",                  "Commercial Services"),
    ("ROL",  "Rollins",                      "Commercial Services"),
    # Professional Services
    ("PAYX", "Paychex",                      "Professional Services"),
    ("ADP",  "Automatic Data Processing",    "Professional Services"),
    ("BR",   "Broadridge Financial",         "Professional Services"),
    ("LDOS", "Leidos Holdings",              "Professional Services"),
    # Ground Transportation
    ("UNP",  "Union Pacific",                "Ground Transportation"),
    ("CSX",  "CSX",                          "Ground Transportation"),
    ("NSC",  "Norfolk Southern",             "Ground Transportation"),
    ("ODFL", "Old Dominion Freight Line",    "Ground Transportation"),
    ("JBHT", "J.B. Hunt Transport",          "Ground Transportation"),
    # Trading & Distribution
    ("URI",  "United Rentals",               "Trading & Distribution"),
    ("FAST", "Fastenal",                     "Trading & Distribution"),
    ("WSO",  "Watsco",                       "Trading & Distribution"),
    ("GWW",  "W.W. Grainger",                "Trading & Distribution"),
]


def build_universe():
    df = pd.DataFrame(INDUSTRIALS, columns=["ticker", "name", "sub_industry"])
    df["sector"] = "Industrials"
    df = df.drop_duplicates(subset="ticker").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_universe()
    print(f"Industrials universe: {len(df)} tickers")
    print("\nBy sub-industry:")
    print(df["sub_industry"].value_counts())
    df.to_csv("/home/claude/quant_industrials/data/raw/universe_industrials.csv", index=False)
    print("\nSaved.")
