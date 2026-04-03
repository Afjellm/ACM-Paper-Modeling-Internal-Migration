library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(sf)
library(flowmapper)
library(mapdeck)
library(purrr)



df <- read_csv("output/all_simulation_results.csv", show_col_types = FALSE) %>%
  mutate(
    year = as.integer(year),
    across(c(delta_gravity, delta_ensemble, delta_amount), ~ suppressWarnings(as.numeric(.))),
    area_code = as.factor(area_code),
    scenario_name = as.factor(scenario_name)
  ) %>%
  # replace area codes with names
  mutate(
    area_code = recode(area_code,
                       `316` = "Mistelbach",
                       `314` = "Lilienfeld",
                       `502` = "Hallein")
  )

df_original <- df %>%
  filter(scenario_name == "original" & area_code == "Hallein")

df_long <- df %>%
  pivot_longer(c(delta_gravity, delta_ensemble, delta_amount, delta_xg, delta_cb),
               names_to = "model", values_to = "delta_value") %>%
  filter(is.finite(year), is.finite(delta_value)) 

results_xg_boost <- df_long %>%
  filter(model == "Constrained XGBoost") %>%
  filter(scenario_name =="baseline") %>%
  filter(area_code == "Lilienfeld") %>%
  arrange(scenario_name, desc(year))

# nicer labels (optional)
df_long$model <- recode(df_long$model,
                        delta_gravity = "Gravity Model",
                        delta_ensemble = "AutoGluon",
                        delta_xg = "Constrained XGBoost",
                        delta_cb = "Constrained Catboost",
                        delta_amount = "Ground Truth")


scenarios <- unique(df_long$scenario_name)

scenario_colors <- setNames(
  scales::hue_pal()(length(scenarios)),
  scenarios
)
scenario_colors["original"] <- "black"

# Define sizes: thicker for 'original'
scenario_sizes <- setNames(rep(0.7, length(scenarios)), scenarios)
scenario_sizes["original"] <- 1.5


area_names <- tibble(
  area_code = factor(c("314", "502")),
  area_name = c("Lilienfeld", "Hallein")
)

df <- df %>%
  left_join(area_names, by = "area_code") %>%
  mutate(area_code = area_name)


plot_1_data <- df_long %>%
  filter(model == "Ground Truth" | 
           model == "Gravity Model" |
           model == "AutoGluon")%>%
  filter(!(model == "Ground Truth" & scenario_name != "original"))


plot_2_data <- df_long %>%
  filter(model == "Ground Truth" | 
           model == "Constrained XGBoost" |
           model == "Constrained Catboost")%>%
  filter(!(model == "Ground Truth" & scenario_name != "original"))


gt_data <- df_long %>%
  filter(model == "Ground Truth", scenario_name == "original")

make_plot <- function(data, models) {
  gt_expanded <- do.call(rbind, lapply(models, function(m) mutate(gt_data, model = m)))
  
  ggplot(data, aes(x = year, y = delta_value, group = scenario_name)) +
    geom_line(aes(color = scenario_name, size = scenario_name), alpha = 0.9) +
    geom_point(aes(color = scenario_name), size = 2) +
    geom_line(data = gt_expanded, aes(linetype = "Ground Truth"), color = "black", linewidth = 1.5) +
    scale_color_manual(values = scenario_colors) +
    scale_size_manual(values = scenario_sizes) +
    scale_linetype_manual(values = c("Ground Truth" = "22"), name = NULL) +
    facet_grid(area_code ~ model, scales = "free_y") +
    theme_minimal(base_size = 20) +
    labs(x = "Year", y = "Net Migration", color = "Scenario", size = "Scenario") +
    theme(strip.text = element_text(face = "bold"))
}

plot_models_1 <- make_plot(
  data   = plot_1_data %>% filter(model != "Ground Truth"),
  models = c("Gravity Model", "AutoGluon")
)

plot_models_2 <- make_plot(
  data   = plot_2_data %>% filter(model != "Ground Truth"),
  models = c("Constrained XGBoost", "Constrained Catboost")
)

plot_models_1
plot_models_2
