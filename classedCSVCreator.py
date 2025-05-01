import pandas as pd
import os, sys, random

def add_column_to_csv(file_path, new_column_name, red_point_ratio):
    try:
        df = pd.read_csv(file_path)

        num_rows = len(df)
        num_reds = int((red_point_ratio/100)*num_rows)
        # red: 0, blue: 1
        new_column_values = [0]*num_reds + [1]*(num_rows-num_reds)
        random.shuffle(new_column_values)
        df[new_column_name] = new_column_values

        # assuming it is a csv file
        outfile_name = f"{file_path[:-4]}{new_column_name}{red_point_ratio}.csv"

        df.to_csv(outfile_name, index=False)
        print(f"New column '{new_column_name}' added. Saved to '{outfile_name}'.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(e)

def main():
    if len(sys.argv) < 3:
        print("Please input sufficient arguments")
        sys.exit()
    file_name = sys.argv[1]
    red_point_ratio = int(sys.argv[2])
    new_column_name = "type"
    file_path = os.path.join('.','data',file_name)
    add_column_to_csv(file_path, new_column_name,red_point_ratio)

if __name__=="__main__":
    main()