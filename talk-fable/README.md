# Lab group talk (Slidev)

    npm install        # once
    npm run dev        # opens the deck; it reloads when a file is saved

- `slides.md` is the talk. One slide per `---` block: `claim` is the slide's title, the body is its visual, and
  the comment at the bottom is what to say (shown in presenter view, at `/presenter`).
- `parts.ts` names the five parts. `styles/index.css` holds every colour and size.
- `components/` are the diagrams drawn in the slides (the balloon, the score bar, the story arc).
- `figures/*.py` draw the plots from the repo's own rows and records into `public/fig/`.
  Rebuild them all with `figures/build.sh`; colours and fonts are in `figures/style.py`.
- `public/poster/` are crops of the summer poster's figures.

A figure that builds up over clicks is saved as layers (`name-0.png`, `name-1.png`, ...) and shown with
`<Layers name="..." :n="..." />`.

`npm run export` writes a PDF with one page per click.
