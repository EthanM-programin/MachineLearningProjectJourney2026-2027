import pandas as pd

def main():
    print("Day 8: Enter Pandas (Data Wrangling)")

    # 1. Creating a mock dataset using a standard Python dictionary
    # Imagine this is data scraped from a local server network
    raw_data = {
        "Server_ID": [101, 102, 103, 104],
        "Traffic_Load_TB": [2.5, 8.1, 1.2, 5.5],
        "Threat_Detected": [False, True, False, True]
    }

    # 2. Converting the raw dictionary into a Pandas DataFrame
    df = pd.DataFrame(raw_data)
    print("\nOur Raw Data Table:")
    print(df)

    # 3. Pulling specific data out (isolating inputs for an AI)
    print("\nTraffic Load Column:")
    print(df["Traffic_Load_TB"])

if __name__ == "__main__":
    main()