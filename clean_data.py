import logging
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
logging.basicConfig(
    level=logging.INFO,  # poziom logowania
    handlers=[
        logging.FileHandler("log.txt", mode='w'),  # zapis do pliku
    ]
)
logging.info("Rozpoczęcie przetwarzania danych")


df = pd.read_csv("data_student_27753.csv")
logging.info(f"Dane wczytane. Liczba wierszy: {df.shape[0]}, kolumn: {df.shape[1]}")

# print(df)
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.shape)
initial_rows = df.shape[0]
initial_cells = df.shape[0] * df.shape[1]
df = df.dropna(subset=['Płeć'])
removed_rows = initial_rows - df.shape[0]
removed_percent = removed_rows / initial_rows * 100

logging.info(f"Liczba usuniętych wierszy (brak płci): {removed_rows}")

# print(df.isnull().sum())
nulls_before = df.isnull().sum()




median_age = int(df['Wiek'].median())
df['Wiek'] = df['Wiek'].fillna(median_age)

df['Wykształcenie'] = df['Wykształcenie'].fillna('Średnie')


mean_salary = df['Średnie Zarobki'].mean()
df['Średnie Zarobki'] = df['Średnie Zarobki'].fillna(mean_salary)


def fill_time_mean(column):
    times = df[column].dropna().str.split(':', expand=True).astype(int)
    mean_hour = int(times[0].mean())
    mean_minute = int(times[1].mean())
    df[column] = df[column].fillna(f"{mean_hour:02}:{mean_minute:02}")

fill_time_mean('Czas Początkowy Podróży')
fill_time_mean('Czas Końcowy Podróży')

df['Cel Podróży'] = df['Cel Podróży'].fillna('Inne')



numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = MinMaxScaler(feature_range=(0, 1))
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


df['Płeć'] = df['Płeć'].map({'Kobieta': 0, 'Mężczyzna': 1})
df['Wykształcenie'] = df['Wykształcenie'].map({'Podstawowe': 0, 'Średnie': 1, 'Wyższe': 2})
df['Cel Podróży'] = df['Cel Podróży'].map({'Edukacja': 0, 'Inne': 1, 'Praca': 2, 'Rozrywka': 3, 'Zakupy': 4})


logging.info(f"Zestandaryzowano kolumny: {list(numeric_cols)}")
logging.info("Zakończenie standaryzacji danych")



# print(df.shape)
# print(df.isnull().sum())
nulls_after = df.isnull().sum()
filled = nulls_before - nulls_after
filled_percent = filled.sum() / (initial_cells-removed_rows*df.shape[1]) * 100
logging.info(f"Liczba wypełnionych danych w kolumnach: \n{filled}")

logging.info(f"Liczba wierszy po czyszczeniu: {df.shape[0]}")
logging.info("Zakończenie przetwarzania danych")

print(f"Procent danych, które zostały zmienione w wyniku uzupełniania braków: {filled_percent:.2f}%")
print(f"Procent danych, które zostały usunięte w wyniku czyszczenia: {removed_percent:.2f}%")
