export default function Methodology() {
  return (
    <div className="prose prose-invert max-w-3xl">
      <h1>Methodology</h1>

      <h2>Dataset</h2>
      <p>
        The classifier is fine-tuned on the <strong>Gravity Spy</strong>{" "}
        training set (Bahaadini et al.), ~10,000 citizen-science-labeled
        LIGO glitches across 22 morphological classes, hosted on Zenodo.
        Out-of-distribution evaluation uses public O4 strain data from{" "}
        <strong>GWOSC</strong>.
      </p>

      <h2>Architecture</h2>
      <p>
        Strain segments are converted to Q-transform spectrograms via{" "}
        <code>gwpy</code> (Q range 4-64, frequency range 10-2048 Hz),
        resized to 224x224, and classified with an EfficientNet-B0
        backbone pretrained on ImageNet and fine-tuned on Gravity Spy.
      </p>

      <h2>Explainability</h2>
      <p>
        Every inference produces a GradCAM overlay computed from the final
        convolutional block, showing which time-frequency regions of the
        spectrogram drove the predicted class.
      </p>

      <h2>Out-of-distribution detection</h2>
      <p>
        We use an energy-based score on the output logits (Liu et al.,
        2020), calibrated against a held-out validation percentile.
        Signals whose energy score exceeds that threshold are flagged as
        potentially novel and surfaced in the Anomaly Gallery. A
        Mahalanobis-distance-on-embeddings variant is available as a
        stricter alternative (<code>model/ood.py:MahalanobisOOD</code>).
      </p>

      <h2>Known limitations</h2>
      <ul>
        <li>Training data is H1/L1-dominated; Virgo and KAGRA are not supported.</li>
        <li>Cross-detector generalization (H1-trained models applied to L1) is weaker and being actively evaluated.</li>
        <li>OOD flags indicate the model's uncertainty, not a confirmed new astrophysical or instrumental phenomenon.</li>
        <li>Raw-CSV sample-rate inference is unreliable on irregularly sampled data.</li>
      </ul>

      <p className="text-sm text-slate-400">
        Code: <a href="#">GitHub repo</a> &middot; Model weights: available for download &middot;
        Writeup: <a href="#">project summary</a>
      </p>
    </div>
  );
}
