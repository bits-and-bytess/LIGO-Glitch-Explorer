# Design notes

Grounding: the tool's own output (Q-transform spectrograms) already has a
built-in signature palette -- viridis, the colormap used throughout
preprocessing/qtransform.py. Gravitational-wave science also has one of
the most recognizable images in modern physics: the GW150914 "chirp"
waveform (a rising-frequency sine sweep). Both are used as the site's
actual visual identity instead of an arbitrary dark-mode-plus-accent
default.

## Color (named viridis stops, not arbitrary picks)
- void        #0c0a16  -- near-black, violet undertone (viridis' dark end, #440154, desaturated)
- panel       #17131f
- panel-raised #201a2c
- hairline    #2c2438
- ink         #ede9f5  -- warm off-white, not pure #fff
- ink-muted   #948da3
- teal        #21918c  -- viridis ~0.4 stop, primary interactive accent
- green       #35b779  -- viridis ~0.6 stop, secondary/success
- yellow      #fde725  -- viridis endpoint, used sparingly for emphasis only
- anomaly     #e8823c  -- deliberately OUTSIDE the viridis family: OOD/alert
                          color reads as "doesn't belong" by design, not
                          just a semantic red/orange convention
- anomaly-bright #f4a261

## Type
- display: Space Grotesk -- geometric, technical, restrained use (headings only)
- sans:    IBM Plex Sans -- built for data-dense technical products
- mono:    IBM Plex Mono -- ALL numeric/technical data (GPS times, Hz,
           confidence %, sample rates, detector codes) gets consistent
           tabular monospace treatment, reinforcing "this is instrument
           data" throughout the whole product, not just in one spot

## Signature elements
1. ChirpMark -- small SVG rising-frequency sine sweep (echoes the actual
   GW150914 waveform shape), used as the site's mark in the nav, and
   reused small in loading/empty states. This is the one deliberately
   memorable element; used with restraint elsewhere.
2. FrequencyAxis divider -- thin horizontal rule with tick marks at
   intervals, echoing a spectrogram's own frequency axis, used as a
   structural section divider instead of a generic <hr>.

## What NOT to do (checked against this explicitly)
- Warm cream + terracotta serif look: not used.
- Near-black + single acid accent: this is what the FIRST pass of this
  frontend looked like (void/panel/accent-teal/signal-indigo) -- flagged
  as the generic AI-dark-mode default and replaced with the above.
- Broadsheet/hairline-rule newspaper look: not used, though the
  FrequencyAxis divider borrows "thin rule" as a content-grounded choice,
  not a decorative one.
