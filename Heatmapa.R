
library(readr)
library(ggcorrplot)
library(ggplot2)

# Funkcja pomocnicza do rysowania heatmapy
generate_heatmap <- function(filepath, title) {
  # Wczytaj dane
  df <- read_csv(filepath)
  
  # Zostaw tylko kolumny numeryczne
  df_numeric <- df[, sapply(df, is.numeric)]
  
  # Macierz korelacji
  corr_matrix <- cor(df_numeric, use = "complete.obs", method = "pearson")
  
  # Heatmapa
  ggcorrplot(corr_matrix, 
             method = "square", 
             type = "lower",
             lab = TRUE,
             title = title,
             colors = c("blue", "white", "red")) +
    theme(plot.title = element_text(hjust = 0.5))
}

# Wygeneruj heatmapy dla 3 zestawów danych
heatmap1 <- generate_heatmap("HistoricData_with_CAQI.csv", "Korelacje – Dane historyczne")
heatmap2 <- generate_heatmap("24HData_with_CAQI.csv", "Korelacje – Ostatnie 24h")
heatmap3 <- generate_heatmap("TodaysData_with_CAQI.csv", "Korelacje – Dzisiaj")
heatmap4 <-generate_heatmap("2023_data_with_CAQI.csv","Korelacje - rok 2023")
heatmap5<-generate_heatmap("merged_energy_pollution_weather.csv","Korelacje - jakość powietrza a pogoda")

print(heatmap1)
print(heatmap2)
print(heatmap3)
print(heatmap4)
print(heatmap5)