
library(mlr3verse)
library(ggplot2)
library(data.table)
library(GGally)


# Wczytanie danych
data_2023 <- fread("2023_data_with_CAQI.csv")

# Przygotowanie danych do analizy
# Wybór cech do analizy
data_2023 <- data_2023[, .(
  `PkRzeszPilsu-PM2.5-1g`,
  `PkRzeszPilsu-PM10-1g`,
  `PkRzeszPilsu-CO-1g`,
  `PkRzeszPilsu-NO2-1g`,
  `PkRzeszRejta-O3-1g`,
  `PkRzeszRejta-SO2-1g`,
  CAQI
)]

# Zamiana nazw kolumn na prostsze
colnames(data_2023) <- c("PM2.5", "PM10", "CO", "NO2", "O3", "SO2", "CAQI")

# Dodaj kolumnę RiskLevel na podstawie CAQI (używając kwantyli)
data_2023[, RiskLevel := cut(CAQI, breaks = quantile(CAQI, probs = c(0, 1/3, 2/3, 1)), labels = c("Low", "Medium", "High"), include.lowest = TRUE)]
data_2023[, RiskLevel := factor(RiskLevel, levels = c("Low", "Medium", "High"))]

# Funkcja do tworzenia histogramów według RiskLevel
hist_with_risk <- function(data, mapping, ...) {
  ggplot(data = data, mapping = mapping) +
    geom_histogram(aes(fill = RiskLevel), bins = 30, position = "identity", alpha = 0.5) +
    scale_fill_manual(values = c("Low" = "green", "Medium" = "blue", "High" = "purple")) +
    theme_minimal()
}

# Funkcja do tworzenia punktów rozproszenia z kolorowaniem
points_with_risk <- function(data, mapping, ...) {
  ggplot(data = data, mapping = mapping) +
    geom_point(aes(color = RiskLevel), alpha = 0.5) +
    scale_color_manual(values = c("Low" = "green", "Medium" = "blue", "High" = "purple")) +
    theme_minimal()
}

# Funkcja do tworzenia boxplotów dla RiskLevel vs zmiennych numerycznych
box_with_risk <- function(data, mapping, ...) {
  ggplot(data = data, mapping = mapping) +
    geom_boxplot(aes(fill = RiskLevel), alpha = 0.7) +
    scale_fill_manual(values = c("Low" = "green", "Medium" = "blue", "High" = "purple")) +
    theme_minimal()
}

# Generowanie wykresu par zmiennych
ggpairs(data_2023[, .(RiskLevel, PM2.5, PM10, CO, NO2, O3, SO2, CAQI)],
        columns = 1:8,
        mapping = aes(color = RiskLevel),
        upper = list(continuous = wrap("cor", size = 3, stars = TRUE), combo = box_with_risk),
        diag = list(continuous = hist_with_risk, discrete = "barDiag"),
        lower = list(continuous = points_with_risk, combo = wrap("dot", alpha = 0.7))) +
  scale_color_manual(values = c("Low" = "green", "Medium" = "blue", "High" = "purple")) +
  scale_fill_manual(values = c("Low" = "green", "Medium" = "blue", "High" = "purple")) +
  ggtitle("Wykres par zmiennych (2023_data_with_CAQI)") +
  theme_minimal() +
  guides(color = guide_legend(title = "RiskLevel"), fill = guide_legend(title = "RiskLevel"))

ggsave("Pairs_plot_2023.png", width = 12, height = 12)