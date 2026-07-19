import FrequencyAxis from "../components/FrequencyAxis";

export default function Methodology() {
  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-display font-semibold mb-2">Methodology</h1>
        <p className="text-ink-muted text-sm">How the model works, and where it's limited.</p>
      </div>

      <FrequencyAxis />

      <Section title="Dataset">
        The classifier is fine-tuned on the <strong className="text-ink">Gravity Spy</strong>{" "}
        training set (Bahaadini et al.), ~10,000 citizen-science-labeled
        LIGO glitches across 22 morphological classes, hosted on Zenodo.
        Out-of-distribution evaluation uses public O4 strain data from{" "}
        <strong className="text-ink">GWOSC</strong>.
      </Section>

      <Section title="Architecture">
        Strain segments are converted to Q-transform spectrograms via{" "}
        <code className="font-mono text-teal">gwpy</code> (Q range 4&ndash;64,
        frequency range 10&ndash;2048 Hz), resized to 224&times;224, and
        classified with an EfficientNet-B0 backbone pretrained on ImageNet
        and fine-tuned on Gravity Spy.
      </Section>

      <Section title="Explainability">
        Every inference produces a GradCAM overlay computed from the final
        convolutional block, showing which time-frequency regions of the
        spectrogram drove the predicted class.
      </Section>

      <Section title="Out-of-distribution detection">
        We use an energy-based score on the output logits, calibrated
        against a held-out validation percentile. Signals whose energy
        score exceeds that threshold are flagged as potentially novel and
        surfaced in the Anomaly Gallery.
      </Section>

      <Section title="Known limitations">
        <ul className="list-disc list-inside space-y-1.5 text-ink-muted">
          <li>Training data is H1/L1-dominated; Virgo and KAGRA are not supported.</li>
          <li>Small-support classes (fewer than ~10 test examples) have noisy per-class metrics almost by construction -- treat those numbers as indicative, not precise.</li>
          <li>Cross-detector generalization (H1-trained models applied to L1) is weaker and being actively evaluated.</li>
          <li>An OOD flag means "doesn't match a known class well," not a confirmed novel astrophysical or instrumental phenomenon.</li>
        </ul>
      </Section>

      <p className="text-sm font-mono text-ink-muted pt-2 border-t border-hairline">
        Code: <a href="#" className="text-teal underline">GitHub repo</a> &middot; Model weights: available for download &middot;
        Writeup: <a href="#" className="text-teal underline">project summary</a>
      </p>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section className="space-y-2">
      <h2 className="text-lg font-display font-medium">{title}</h2>
      <div className="text-ink-muted leading-relaxed">{children}</div>
    </section>
  );
}
