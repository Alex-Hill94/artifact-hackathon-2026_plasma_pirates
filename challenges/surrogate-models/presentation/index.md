---
title: "Plasma Pirates: Surrogate Models of the PALLAS Simulation"
author: "Hayg Guler · Alexander Hill · Sébastien Joly · Keon Hee Kim · Mykyta Lenivenko · Guillermo Rodrîguez Llorente"
format:
  revealjs:
    theme: [default, custom.scss]
    slide-number: c/t
    show-slide-number: all
    hash: true
    controls: true
    progress: true
    transition: slide
    background-transition: fade
    width: 1280
    height: 720
    margin: 0.05
    center: false
    footer: "AISSAI Hackathon 2026 · Plasma Pirates"
    include-after-body: _afterbody.html
execute:
  echo: false
  eval: false
  warning: false
  message: false
---

{{< include sections/_01-why-surrogates.md >}}

{{< include sections/_02-pallas-overview.md >}}

{{< include sections/_03-simulation-to-surrogate.md >}}

{{< include sections/_04-data-exploration.md >}}

{{< include sections/_05-read-hopsworks.md >}}

{{< include sections/_06-train-register.md >}}

{{< include sections/_07-modelling-directions.md >}}

{{< include sections/_08-data-quality.md >}}

{{< include sections/_09-trajectory-outliers.md >}}

{{< include sections/_10-plotly-scatter.md >}}

{{< include sections/_11-model-experiments.md >}}

{{< include sections/_12-results.md >}}

{{< include sections/_13-takeaways.md >}}

{{< include sections/_14-questions.md >}}
