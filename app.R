# =============================================================================
# Airbnb NLP + Data Visualization — Interactive R Shiny App
# MongoDB Atlas → data wrangling → 4+ interactive plotly visuals +
# 3 text-mining frameworks (TF-IDF, AFINN Sentiment, LDA Topic Modeling)
# =============================================================================

# --- Package management -------------------------------------------------------
required_pkgs <- c(
  "shiny", "bslib", "mongolite", "dplyr", "tidyr", "ggplot2", "plotly",
  "forcats", "purrr", "stringr", "tidytext", "textdata", "topicmodels",
  "reshape2", "shinycssloaders", "DT", "scales"
)
missing_pkgs <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
if (length(missing_pkgs) > 0) install.packages(missing_pkgs)
invisible(lapply(required_pkgs, library, character.only = TRUE))

# Robust AFINN fetch: try tidytext first, fall back to direct download, then bing
get_afinn <- function() {
  # Attempt 1: tidytext/textdata cache (works if user already accepted prompt)
  lex <- tryCatch(suppressWarnings(get_sentiments("afinn")), error = function(e) NULL)
  if (!is.null(lex) && is.data.frame(lex) && nrow(lex) > 0) return(lex)
  # Attempt 2: direct download from source
  res <- tryCatch({
    tmp <- tempfile(fileext = ".txt")
    download.file("https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt",
                  tmp, quiet = TRUE, mode = "w")
    df <- read.delim(tmp, header = FALSE, col.names = c("word", "value"),
                     stringsAsFactors = FALSE, quote = "")
    if (nrow(df) > 100) tibble(word = df$word, value = as.integer(df$value)) else NULL
  }, error = function(e2) NULL)
  if (!is.null(res)) return(res)
  # Attempt 3: bing sentiment with numeric mapping (always available)
  bing <- get_sentiments("bing")
  bing %>% mutate(value = ifelse(sentiment == "positive", 1L, -1L)) %>% select(word, value)
}
afinn_lexicon <- get_afinn()
message("AFINN lexicon loaded: ", nrow(afinn_lexicon), " words")

# =============================================================================
# THEME + STYLE
# =============================================================================
# bslib ships with Google Fonts via CDN — works in browser without local install
app_theme <- bs_theme(

  version   = 5,
  bootswatch = "darkly",
  base_font  = font_google("Inter"),
  heading_font = font_google("Inter"),
  bg = "#0f172a", fg = "#e2e8f0",
  primary = "#38bdf8", secondary = "#a855f7",
  "card-bg" = "#111827"
)

# Custom CSS for glassmorphism cards, smooth transitions, and spacing
custom_css <- tags$style(HTML("
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  body { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }
  .kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 16px;
  }
  .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(56,189,248,0.15); }
  .kpi-label { color: #64748b; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin: 0; }
  .kpi-value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin: 6px 0 4px; }
  .kpi-note  { color: #475569; font-size: 0.75rem; margin: 0; }
  .chart-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
  }
  .nav-tabs .nav-link { color: #94a3b8 !important; font-weight: 500; border: none !important; }
  .nav-tabs .nav-link.active { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; background: transparent !important; }
  .sidebar { border-right: 1px solid #1e293b; }
  .selectize-input, .selectize-dropdown { background: #1e293b !important; color: #e2e8f0 !important; border-color: #334155 !important; }
  .selectize-input .item { color: #e2e8f0 !important; }
  .irs--shiny .irs-bar { background: #38bdf8; }
  .irs--shiny .irs-handle { border-color: #38bdf8; background: #38bdf8; }
  .irs--shiny .irs-from, .irs--shiny .irs-to, .irs--shiny .irs-single { background: #38bdf8; }
  .section-header { color: #94a3b8; font-size: 0.9rem; font-weight: 500; margin-bottom: 12px; }
  h5.tab-title { color: #e2e8f0; font-weight: 600; margin-bottom: 4px; }
  .plotly .main-svg { background: transparent !important; }
  #loading-overlay {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: #0f172a; z-index: 9999;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    transition: opacity 0.5s ease;
  }
  #loading-overlay.fade-out { opacity: 0; pointer-events: none; }
  .spinner-ring { animation: spin 1.2s cubic-bezier(0.5,0,0.5,1) infinite; transform-origin: 40px 40px; }
  .spinner-ring:nth-child(1) { animation-delay: -0.45s; }
  .spinner-ring:nth-child(2) { animation-delay: -0.3s; }
  .spinner-ring:nth-child(3) { animation-delay: -0.15s; }
  @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  .loading-text { color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 0.95rem; font-weight: 400; margin-top: 24px; letter-spacing: 0.02em; }
  .loading-sub { color: #475569; font-family: 'Inter', sans-serif; font-size: 0.75rem; margin-top: 8px; }
"))

# Plotly layout for consistent dark styling
plotly_layout <- function(p, title = "", xlab = "", ylab = "", showlegend = TRUE) {
  p %>% layout(
    title = list(text = title, font = list(color = "#e2e8f0", size = 15, family = "Inter")),
    xaxis = list(title = list(text = xlab, font = list(color = "#94a3b8", size = 12)),
                 tickfont = list(color = "#64748b", size = 11), gridcolor = "#1e293b",
                 zerolinecolor = "#1e293b"),
    yaxis = list(title = list(text = ylab, font = list(color = "#94a3b8", size = 12)),
                 tickfont = list(color = "#64748b", size = 11), gridcolor = "#1e293b",
                 zerolinecolor = "#1e293b"),
    paper_bgcolor = "transparent", plot_bgcolor = "transparent",
    font = list(family = "Inter", color = "#e2e8f0"),
    legend = list(font = list(color = "#94a3b8", size = 11),
                  bgcolor = "transparent", orientation = "h",
                  x = 0.5, xanchor = "center", y = -0.15),
    showlegend = showlegend,
    margin = list(t = 50, b = 60, l = 60, r = 20),
    hoverlabel = list(bgcolor = "#1e293b", font = list(color = "#e2e8f0", family = "Inter"))
  )
}

# Color palette
pal <- c("Entire home/apt" = "#38bdf8", "Private room" = "#a855f7",
         "Shared room" = "#34d399", "Hotel room" = "#fb923c", "Unknown" = "#64748b")

# =============================================================================
# MONGODB CONNECTION
# =============================================================================
mongo_url <- Sys.getenv(
  "AIRBNB_MONGO_URL",
  unset = "mongodb+srv://testuser90000000_db_user:AW7kn2kDbqlN3qNd@cluster0.rnhn6aj.mongodb.net/?retryWrites=true&w=majority"
)

field_spec <- paste0(
  '{"_id":1,"name":1,"description":1,"address":1,"room_type":1,',
  '"property_type":1,"beds":1,"bedrooms":1,"bathrooms":1,"price":1,',
  '"minimum_nights":1,"number_of_reviews":1,"review_scores":1,',
  '"amenities":1,"host":1,"availability_365":1}'
)

fetch_listings <- function(limit = 5555) {
  if (identical(mongo_url, "") || grepl("<<", mongo_url, fixed = TRUE)) return(NULL)
  col <- mongo(collection = "listingsAndReviews", db = "sample_airbnb", url = mongo_url)
  res <- tryCatch(col$find("{}", fields = field_spec, limit = limit),
                  error = function(e) NULL)
  if (is.null(res) || nrow(res) == 0) return(NULL)
  res
}

# =============================================================================
# ROBUST TYPE CONVERTERS (handles Decimal128 data frames, NULLs, wrong lengths)
# =============================================================================
to_numeric <- function(x, n) {
  if (is.null(x) || length(x) == 0) return(rep(NA_real_, n))
  if (is.data.frame(x)) {
    v <- suppressWarnings(as.numeric(as.character(x[[1]])))
    if (length(v) == n) return(v) else return(rep(NA_real_, n))
  }
  if (is.list(x)) {
    v <- suppressWarnings(as.numeric(vapply(x, function(el) {
      if (is.data.frame(el)) as.character(el[[1]][1])
      else as.character(el)
    }, character(1))))
    if (length(v) == n) return(v) else return(rep(NA_real_, n))
  }
  v <- suppressWarnings(as.numeric(as.character(x)))
  if (length(v) == n) v else rep(NA_real_, n)
}

to_character <- function(x, n) {
  if (is.null(x) || length(x) == 0) return(rep(NA_character_, n))
  if (is.data.frame(x)) { v <- as.character(x[[1]]); if (length(v) == n) return(v) }
  if (is.list(x) && !is.data.frame(x)) {
    v <- vapply(x, function(el) if (is.null(el) || length(el) == 0) NA_character_ else as.character(el[[1]]), character(1))
    if (length(v) == n) return(v)
  }
  v <- as.character(x)
  if (length(v) == n) v else rep(NA_character_, n)
}

nested_col <- function(df, col, n) {
  if (is.null(df) || !is.data.frame(df) || !col %in% names(df)) return(rep(NA_character_, n))
  v <- as.character(df[[col]])
  if (length(v) == n) v else rep(NA_character_, n)
}

nested_num <- function(df, col, n) suppressWarnings(as.numeric(nested_col(df, col, n)))

# =============================================================================
# DATA WRANGLING
# =============================================================================
prep_data <- function(df) {
  if (is.null(df) || nrow(df) == 0) return(NULL)
  n <- nrow(df)

  market_vec    <- nested_col(df$address, "market", n)
  country_vec   <- nested_col(df$address, "country", n)
  suburb_vec    <- nested_col(df$address, "suburb", n)
  govt_area_vec <- nested_col(df$address, "government_area", n)
  superhost_vec <- nested_col(df$host, "host_is_superhost", n)
  rating_vec    <- nested_num(df$review_scores, "review_scores_rating", n)

  out <- tibble(
    id            = to_character(df$`_id`, n),
    name          = to_character(df$name, n),
    description   = to_character(df$description, n),
    room_type     = to_character(df$room_type, n),
    property_type = to_character(df$property_type, n),
    market        = ifelse(is.na(market_vec) | market_vec == "", country_vec, market_vec),
    neighborhood  = ifelse(is.na(suburb_vec) | suburb_vec == "", govt_area_vec, suburb_vec),
    superhost     = superhost_vec %in% c("true", "t", "TRUE", "True"),
    price         = to_numeric(df$price, n),
    beds          = to_numeric(df$beds, n),
    bedrooms      = to_numeric(df$bedrooms, n),
    bathrooms     = to_numeric(df$bathrooms, n),
    min_nights    = to_numeric(df$minimum_nights, n),
    num_reviews   = to_numeric(df$number_of_reviews, n),
    rating        = rating_vec,
    availability  = to_numeric(df$availability_365, n),
    amenities_raw = if (!is.null(df$amenities)) df$amenities else as.list(rep(NA_character_, n))
  )

  out$amenities <- vapply(out$amenities_raw, function(x) {
    if (is.null(x) || length(x) == 0) NA_character_ else paste(x, collapse = ", ")
  }, character(1))
  out$amenity_count <- vapply(out$amenities_raw, function(x) {
    if (is.null(x)) 0L else length(x)
  }, integer(1))
  out$amenities_raw <- NULL

  out %>%
    mutate(
      price_per_bed = if_else(!is.na(price) & beds > 0, price / beds, NA_real_),
      desc_len      = nchar(description),
      room_type     = factor(ifelse(is.na(room_type), "Unknown", room_type)),
      market        = factor(ifelse(is.na(market) | market == "", "Unknown", market)),
      neighborhood  = factor(ifelse(is.na(neighborhood) | neighborhood == "", "Unknown", neighborhood))
    ) %>%
    filter(price > 0, price < quantile(price, 0.99, na.rm = TRUE)) %>%
    filter(!is.na(description), description != "")
}

# =============================================================================
# TEXT MINING HELPERS
# =============================================================================
tokenize_descriptions <- function(df) {
  df %>%
    select(id, room_type, market, price, rating, num_reviews, description) %>%
    unnest_tokens(word, description) %>%
    anti_join(stop_words, by = "word") %>%
    filter(str_detect(word, "^[a-z]{2,}$"))
}

# Framework 1 — TF-IDF
tfidf_by_group <- function(tokens, group_var = "room_type", top_n = 10) {
  tokens %>%
    count(!!sym(group_var), word, sort = TRUE) %>%
    bind_tf_idf(word, !!sym(group_var), n) %>%
    group_by(!!sym(group_var)) %>%
    slice_max(tf_idf, n = top_n, with_ties = FALSE) %>%
    ungroup()
}

# Framework 2 — AFINN Sentiment
sentiment_by_group <- function(tokens) {
  tokens %>%
    inner_join(afinn_lexicon, by = "word") %>%
    group_by(room_type) %>%
    summarize(avg_sentiment = mean(value, na.rm = TRUE),
              pos_words = sum(value > 0), neg_words = sum(value < 0),
              n_words = n(), .groups = "drop")
}

# Framework 3 — LDA Topic Modeling
topic_terms_fn <- function(tokens, k = 5, top_n = 6) {
  dtm <- tokens %>% count(id, word) %>% cast_dtm(id, word, n)
  lda_model <- LDA(dtm, k = k, control = list(seed = 42))
  tidy(lda_model, matrix = "beta") %>%
    group_by(topic) %>%
    slice_max(beta, n = top_n, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(topic_label = paste("Topic", topic))
}

# =============================================================================
# UI
# =============================================================================
loading_overlay <- tags$div(id = "loading-overlay",
  HTML('
    <svg width="80" height="80" viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg">
      <circle class="spinner-ring" cx="40" cy="40" r="32" fill="none" stroke="#38bdf8" stroke-width="3" stroke-linecap="round" stroke-dasharray="24 176" />
      <circle class="spinner-ring" cx="40" cy="40" r="24" fill="none" stroke="#a855f7" stroke-width="3" stroke-linecap="round" stroke-dasharray="18 132" />
      <circle class="spinner-ring" cx="40" cy="40" r="16" fill="none" stroke="#34d399" stroke-width="3" stroke-linecap="round" stroke-dasharray="12 88" />
    </svg>'),
  tags$div(class = "loading-text", "Loading Airbnb data from MongoDB Atlas..."),
  tags$div(class = "loading-sub", "Fetching listings, reviews & text data")
)

ui <- page_sidebar(
  theme = app_theme,
  title = div(
    style = "display:flex; align-items:center; gap:10px;",
    tags$span("\U0001F3E0", style = "font-size:1.4rem;"),
    tags$span("Airbnb Insights", style = "font-weight:700; font-size:1.2rem;"),
    tags$span("Text Mining & Visualization",
              style = "font-weight:300; color:#64748b; font-size:0.9rem; margin-left:4px;")
  ),
  sidebar = sidebar(
    width = 280,
    bg = "#111827",
    custom_css,
    tags$p("Filter listings to explore pricing, reviews, and language patterns.",
           class = "section-header"),
    hr(style = "border-color:#1e293b; margin:8px 0;"),
    sliderInput("price_range", "Price range (USD)",
                min = 0, max = 10000, value = c(0, 10000), step = 10),
    selectInput("room_filter", "Room type", choices = NULL, multiple = TRUE),
    selectInput("market_filter", "Market", choices = NULL, multiple = TRUE),
    sliderInput("min_reviews", "Minimum reviews", min = 0, max = 100, value = 0, step = 1),
    hr(style = "border-color:#1e293b; margin:8px 0;"),
    tags$small("Source: MongoDB Atlas  \u00b7  sample_airbnb", style = "color:#475569;"),
    tags$div(style = "margin-top:16px; text-align:center;",
      tags$small(style = "color:#64748b; font-size:0.75rem;",
        "Created by ",
        tags$a(href = "https://www.adityagaurav.com", target = "_blank",
               style = "color:#38bdf8; text-decoration:none; font-weight:500;",
               "Aditya Gaurav")
      )
    )
  ),

  loading_overlay,
  navset_card_tab(
    id = "tabs",
    # ── Tab 1: Dashboard ──────────────────────────────────────────────────────
    nav_panel("Dashboard", icon = icon("chart-line"),
      uiOutput("kpi_cards"),
      layout_columns(
        col_widths = c(6, 6),
        div(class = "chart-card", withSpinner(plotlyOutput("plot_price_rating", height = "370px"), type = 8, color = "#38bdf8", size = 0.6)),
        div(class = "chart-card", withSpinner(plotlyOutput("plot_price_per_bed", height = "370px"), type = 8, color = "#a855f7", size = 0.6))
      )
    ),
    # ── Tab 2: Pricing & Reviews ──────────────────────────────────────────────
    nav_panel("Pricing & Reviews", icon = icon("dollar-sign"),
      layout_columns(
        col_widths = c(6, 6),
        div(class = "chart-card", withSpinner(plotlyOutput("plot_market_price", height = "400px"), type = 8, color = "#38bdf8", size = 0.6)),
        div(class = "chart-card", withSpinner(plotlyOutput("plot_amenity_price", height = "400px"), type = 8, color = "#a855f7", size = 0.6))
      )
    ),
    # ── Tab 3: Text Insights ─────────────────────────────────────────────────
    nav_panel("Text Insights", icon = icon("language"),
      tags$p("Three NLP frameworks applied to listing descriptions", class = "section-header"),
      div(class = "chart-card", withSpinner(plotlyOutput("plot_tfidf", height = "380px"), type = 8, color = "#34d399", size = 0.6)),
      layout_columns(
        col_widths = c(6, 6),
        div(class = "chart-card", withSpinner(plotlyOutput("plot_sentiment", height = "360px"), type = 8, color = "#38bdf8", size = 0.6)),
        div(class = "chart-card", withSpinner(plotlyOutput("plot_topics", height = "360px"), type = 8, color = "#a855f7", size = 0.6))
      )
    ),
    # ── Tab 4: Data Explorer ─────────────────────────────────────────────────
    nav_panel("Data Explorer", icon = icon("table"),
      div(class = "chart-card", withSpinner(DTOutput("data_table"), type = 8, color = "#38bdf8", size = 0.6))
    ),
    # ── Tab 5: Design Philosophy ─────────────────────────────────────────────
    nav_panel("Design Philosophy", icon = icon("lightbulb"),
      div(class = "chart-card", style = "max-width:860px; margin:0 auto; padding:32px;",
        tags$h4("Why I Built It This Way", style = "color:#e2e8f0; font-weight:700; margin-bottom:4px;"),
        tags$p("Consumer-first design, shaped by building Knowasiak.", style = "color:#64748b; font-size:0.85rem; margin-bottom:24px;"),

        tags$div(style = "display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px;",
          # Card 1
          tags$div(class = "kpi-card", style = "text-align:left; padding:20px;",
            tags$div(icon("users"), style = "color:#38bdf8; font-size:1.1rem; margin-bottom:8px;"),
            tags$h6("User-First, Always", style = "color:#e2e8f0; font-weight:600; margin:0 0 6px;"),
            tags$p("Building Knowasiak \u2014 a business social network \u2014 taught me that every design decision must start with the end user. Filters, sliders, and multi-select dropdowns here give users full control over what they see, not what I think they should see.",
                   style = "color:#94a3b8; font-size:0.82rem; line-height:1.5; margin:0;")
          ),
          # Card 2
          tags$div(class = "kpi-card", style = "text-align:left; padding:20px;",
            tags$div(icon("spinner"), style = "color:#a855f7; font-size:1.1rem; margin-bottom:8px;"),
            tags$h6("Visual Feedback at Every Step", style = "color:#e2e8f0; font-weight:600; margin:0 0 6px;"),
            tags$p("At Knowasiak, I learned that silence frustrates users. That is why this dashboard has a loading overlay on launch, per-chart spinners on tab switches, and hover tooltips on every data point \u2014 the interface always communicates its state.",
                   style = "color:#94a3b8; font-size:0.82rem; line-height:1.5; margin:0;")
          ),
          # Card 3
          tags$div(class = "kpi-card", style = "text-align:left; padding:20px;",
            tags$div(icon("hand-pointer"), style = "color:#34d399; font-size:1.1rem; margin-bottom:8px;"),
            tags$h6("Interaction Over Decoration", style = "color:#e2e8f0; font-weight:600; margin:0 0 6px;"),
            tags$p("Every chart is interactive: zoom, pan, hover for detail, click legends to toggle series. Usability is prioritised over aesthetics alone. A beautiful chart you cannot explore is just a picture.",
                   style = "color:#94a3b8; font-size:0.82rem; line-height:1.5; margin:0;")
          ),
          # Card 4
          tags$div(class = "kpi-card", style = "text-align:left; padding:20px;",
            tags$div(icon("universal-access"), style = "color:#fb923c; font-size:1.1rem; margin-bottom:8px;"),
            tags$h6("Usability & Accessibility", style = "color:#e2e8f0; font-weight:600; margin:0 0 6px;"),
            tags$p("High contrast dark theme, clear typographic hierarchy, and consistent colour coding across all charts reduce cognitive load. Lessons from scaling Knowasiak\u2019s interface for diverse users directly informed these choices.",
                   style = "color:#94a3b8; font-size:0.82rem; line-height:1.5; margin:0;")
          )
        ),

        tags$hr(style = "border-color:#1e293b; margin:20px 0;"),
        tags$p(style = "color:#64748b; font-size:0.8rem; line-height:1.6; margin:0;",
          "This dashboard is not styled for the sake of styling. Every element \u2014 the KPI cards, the sidebar filters, ",
          "the plotly interactivity, the loading spinners \u2014 exists because real product experience at ",
          tags$a(href = "https://www.knowasiak.com", target = "_blank",
                 style = "color:#38bdf8; text-decoration:none; font-weight:500;", "Knowasiak"),
          " showed me that consumer satisfaction comes from control, feedback, and clarity."
        ),
        tags$p(style = "color:#475569; font-size:0.75rem; margin-top:12px;",
          "\u2014 Aditya Gaurav \u00b7 ",
          tags$a(href = "https://www.adityagaurav.com", target = "_blank",
                 style = "color:#38bdf8; text-decoration:none;", "adityagaurav.com")
        )
      )
    )
  )
)

# =============================================================================
# SERVER
# =============================================================================
server <- function(input, output, session) {

  raw_data <- reactiveVal(NULL)

  observe({
    df <- prep_data(fetch_listings())
    raw_data(df)
    if (is.null(df)) {
      showNotification("No data loaded — check MongoDB.", type = "error", duration = NULL)
      return()
    }
    price_min <- unname(floor(min(df$price, na.rm = TRUE)))
    price_max <- unname(ceiling(quantile(df$price, 0.98, na.rm = TRUE)))
    updateSelectInput(session, "room_filter",
                      choices = levels(df$room_type), selected = levels(df$room_type))
    updateSelectInput(session, "market_filter",
                      choices = levels(df$market),
                      selected = levels(df$market)[seq_len(min(5, nlevels(df$market)))])
    updateSliderInput(session, "price_range", min = price_min, max = price_max,
                      value = c(price_min, price_max))
    # Dismiss loading overlay
    session$sendCustomMessage("hide-loader", list())
  })

  filtered <- reactive({
    df <- raw_data(); req(df)
    df %>% filter(
      price >= input$price_range[1], price <= input$price_range[2],
      num_reviews >= input$min_reviews,
      if (length(input$room_filter) > 0) room_type %in% input$room_filter else TRUE,
      if (length(input$market_filter) > 0) market %in% input$market_filter else TRUE
    )
  })

  # ── KPI cards ──────────────────────────────────────────────────────────────
  output$kpi_cards <- renderUI({
    df <- filtered(); req(nrow(df) > 0)
    make_kpi <- function(icon_name, label, value, note, accent = "#38bdf8") {
      div(class = "kpi-card",
        tags$div(icon(icon_name), style = paste0("color:", accent, "; font-size:1.3rem; margin-bottom:4px;")),
        tags$p(label, class = "kpi-label"),
        tags$p(value, class = "kpi-value"),
        tags$p(note, class = "kpi-note")
      )
    }
    layout_columns(col_widths = c(3, 3, 3, 3),
      make_kpi("building",    "Listings",   scales::comma(nrow(df)), "After filters", "#38bdf8"),
      make_kpi("dollar-sign", "Avg Price",  dollar(mean(df$price, na.rm = TRUE)), "Per night", "#a855f7"),
      make_kpi("star",        "Avg Rating", round(mean(df$rating, na.rm = TRUE), 1), "Out of 100", "#34d399"),
      make_kpi("user-check",  "Superhosts", paste0(round(100 * mean(df$superhost, na.rm = TRUE), 1), "%"), "Of filtered hosts", "#fb923c")
    )
  })

  # ── Viz 1: Price vs Rating (interactive scatter) ───────────────────────────
  output$plot_price_rating <- renderPlotly({
    df <- filtered(); req(nrow(df) > 5)
    p <- ggplot(df, aes(rating, price, color = room_type,
                        text = paste0(name, "\n", room_type, " · $", price, "/night\nRating: ", rating))) +
      geom_point(alpha = 0.45, size = 1.8) +
      geom_smooth(method = "loess", se = FALSE, linewidth = 0.9) +
      scale_color_manual(values = pal) +
      scale_y_continuous(labels = dollar) +
      labs(x = "Review Score", y = "Price (USD)", color = NULL) +
      theme_void() + theme(legend.position = "none")
    ggplotly(p, tooltip = "text") %>%
      plotly_layout(title = "Price vs Review Rating", xlab = "Review Score (0–100)",
                    ylab = "Price per Night (USD)", showlegend = TRUE)
  })

  # ── Viz 2: Price per bed (box plot) ────────────────────────────────────────
  output$plot_price_per_bed <- renderPlotly({
    df <- filtered() %>% filter(!is.na(price_per_bed)); req(nrow(df) > 5)
    cap <- unname(quantile(df$price_per_bed, 0.95, na.rm = TRUE))
    plot_ly(df %>% filter(price_per_bed <= cap),
            y = ~room_type, x = ~price_per_bed, color = ~room_type,
            colors = pal, type = "box", boxmean = TRUE,
            hoverinfo = "x+y") %>%
      plotly_layout(title = "Price per Bed Distribution",
                    xlab = "Price per Bed (USD)", ylab = "", showlegend = FALSE)
  })

  # ── Viz 3: Market price comparison ─────────────────────────────────────────
  output$plot_market_price <- renderPlotly({
    df <- filtered(); req(nrow(df) > 5)
    top <- df %>% count(market, sort = TRUE) %>% slice_head(n = 10) %>% pull(market)
    agg <- df %>% filter(market %in% top) %>%
      group_by(market) %>%
      summarize(avg_price = mean(price, na.rm = TRUE),
                avg_reviews = round(mean(num_reviews, na.rm = TRUE), 1),
                n = n(), .groups = "drop") %>%
      arrange(avg_price) %>%
      mutate(market = factor(as.character(market), levels = as.character(market)))
    plot_ly(agg, y = ~market, x = ~avg_price,
            type = "bar", orientation = "h",
            marker = list(color = ~avg_reviews,
                          colorscale = list(c(0, "#38bdf8"), c(1, "#a855f7")),
                          showscale = TRUE,
                          colorbar = list(title = "Avg\nReviews")),
            text = ~paste0("$", round(avg_price), " · ", n, " listings · ", avg_reviews, " avg reviews"),
            hoverinfo = "text") %>%
      plotly_layout(title = "Average Price by Market (Top 10)",
                    xlab = "Avg Nightly Price (USD)", ylab = "")
  })

  # ── Viz 4: Amenity count vs price ──────────────────────────────────────────
  output$plot_amenity_price <- renderPlotly({
    df <- filtered(); req(nrow(df) > 5)
    agg <- df %>%
      mutate(amenity_bin = cut(amenity_count,
                               breaks = c(0, 5, 10, 15, 20, 30, Inf),
                               labels = c("1-5", "6-10", "11-15", "16-20", "21-30", "30+"), right = TRUE)) %>%
      group_by(amenity_bin) %>%
      summarize(avg_price = mean(price, na.rm = TRUE), n = n(), .groups = "drop")
    plot_ly(agg, x = ~amenity_bin, y = ~avg_price, type = "bar",
            marker = list(color = ~n, colorscale = list(c(0, "#38bdf8"), c(1, "#a855f7")),
                          showscale = TRUE, colorbar = list(title = "Listings")),
            text = ~paste0("$", round(avg_price), " · ", n, " listings"), hoverinfo = "text") %>%
      plotly_layout(title = "More Amenities = Higher Price?",
                    xlab = "Number of Amenities", ylab = "Avg Price (USD)")
  })

  # ── Text: TF-IDF ──────────────────────────────────────────────────────────
  output$plot_tfidf <- renderPlotly({
    df <- filtered(); req(nrow(df) > 10)
    tokens <- tokenize_descriptions(df); req(nrow(tokens) > 50)
    tfidf <- tfidf_by_group(tokens, "room_type", top_n = 8)
    p <- ggplot(tfidf, aes(tf_idf, fct_reorder(word, tf_idf), fill = room_type)) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~room_type, scales = "free") +
      scale_fill_manual(values = pal) +
      labs(x = "TF-IDF Score", y = NULL) +
      theme_void() + theme(
        strip.text = element_text(color = "#e2e8f0", size = 11, face = "bold"),
        axis.text.y = element_text(color = "#94a3b8", size = 10)
      )
    ggplotly(p, tooltip = c("x", "y")) %>%
      plotly_layout(title = "TF-IDF: Distinctive Words by Room Type",
                    xlab = "TF-IDF Score", ylab = "", showlegend = FALSE)
  })

  # ── Text: Sentiment ───────────────────────────────────────────────────────
  output$plot_sentiment <- renderPlotly({
    df <- filtered(); req(nrow(df) > 10)
    tokens <- tokenize_descriptions(df); req(nrow(tokens) > 50)
    senti <- sentiment_by_group(tokens)
    plot_ly(senti, y = ~fct_reorder(room_type, avg_sentiment), x = ~avg_sentiment,
            type = "bar", orientation = "h",
            marker = list(color = ~avg_sentiment,
                          colorscale = list(c(0, "#ef4444"), c(0.5, "#64748b"), c(1, "#34d399"))),
            text = ~paste0(room_type, "\nSentiment: ", round(avg_sentiment, 2),
                           "\nPos words: ", pos_words, " · Neg words: ", neg_words),
            hoverinfo = "text") %>%
      plotly_layout(title = "Sentiment by Room Type (AFINN)",
                    xlab = "Avg Sentiment Score", ylab = "", showlegend = FALSE)
  })

  # ── Text: LDA Topics ─────────────────────────────────────────────────────
  output$plot_topics <- renderPlotly({
    df <- filtered(); req(nrow(df) > 10)
    tokens <- tokenize_descriptions(df); req(nrow(tokens) > 50)
    terms <- topic_terms_fn(tokens, k = 5, top_n = 6)
    p <- ggplot(terms, aes(beta, fct_reorder(term, beta), fill = topic_label)) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~topic_label, scales = "free") +
      scale_fill_manual(values = c("#38bdf8", "#a855f7", "#34d399", "#fb923c", "#f43f5e")) +
      labs(x = "Beta", y = NULL) +
      theme_void() + theme(
        strip.text = element_text(color = "#e2e8f0", size = 10, face = "bold"),
        axis.text.y = element_text(color = "#94a3b8", size = 9)
      )
    ggplotly(p, tooltip = c("x", "y")) %>%
      plotly_layout(title = "LDA Topic Model: Latent Themes",
                    xlab = "Word Probability (Beta)", ylab = "", showlegend = FALSE)
  })

  # ── Data table ────────────────────────────────────────────────────────────
  output$data_table <- renderDT({
    df <- filtered(); req(df)
    datatable(
      df %>% select(name, market, neighborhood, room_type, price, beds, bedrooms,
                    bathrooms, num_reviews, rating, amenity_count, superhost),
      options = list(pageLength = 20, scrollX = TRUE,
                     initComplete = JS("function(settings, json) {",
                       "$(this.api().table().header()).css({'background-color':'#1e293b','color':'#e2e8f0'});",
                       "$(this.api().table().body()).css({'background-color':'#0f172a','color':'#e2e8f0'});",
                       "}")),
      rownames = FALSE
    ) %>% formatCurrency("price", "$") %>% formatRound("rating", 1)
  })
}

# Inject JS to hide loading overlay on custom message
ui <- tagList(
  ui,
  tags$script(HTML("
    Shiny.addCustomMessageHandler('hide-loader', function(msg) {
      var el = document.getElementById('loading-overlay');
      if (el) { el.classList.add('fade-out'); setTimeout(function(){ el.remove(); }, 600); }
    });
  "))
)

shinyApp(ui, server)
