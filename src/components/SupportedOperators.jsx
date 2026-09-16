import React, { useRef, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import '../styles/modelCompatibility.css';

/** Splits "Property : Value" and returns [property, value]; handles lines without colon. */
const parseSpecLine = (line) => {
  const idx = line.indexOf(' : ');
  if (idx === -1) return [null, line];
  return [line.slice(0, idx).trim(), line.slice(idx + 3).trim()];
};

/**
 * Parses a line that may have subpoints:
 * - "Property : Option1 & Option2 : Detail" (one common detail)
 * - "Property : Option1 : Detail1 & Option2 : Detail2" (per-option details, e.g. Axis)
 * Returns { type: 'simple', property, value } or { type: 'subpoints', property, subpoints: [{ option, detail }] }.
 */
const parseSpecLineWithSubpoints = (line) => {
  const idx = line.indexOf(' : ');
  if (idx === -1) return { type: 'simple', property: null, value: line };
  const property = line.slice(0, idx).trim();
  const valueStr = line.slice(idx + 3).trim();
  if (valueStr.includes(' & ')) {
    const segments = valueStr.split(/\s*&\s*/).map((s) => s.trim()).filter(Boolean);
    const allHaveColon = segments.every((s) => s.includes(' : '));
    if (allHaveColon) {
      const subpoints = segments.map((seg) => {
        const ci = seg.indexOf(' : ');
        return { option: seg.slice(0, ci).trim(), detail: seg.slice(ci + 3).trim() };
      });
      return { type: 'subpoints', property, subpoints };
    }
    const lastColon = valueStr.lastIndexOf(' : ');
    const hasDetail = lastColon > 0;
    const optionsPart = hasDetail ? valueStr.slice(0, lastColon).trim() : valueStr;
    const detailPart = hasDetail ? valueStr.slice(lastColon + 3).trim() : '';
    const options = optionsPart.split(/\s*&\s*/).map((o) => o.trim()).filter(Boolean);
    const subpoints = options.map((option) => ({ option, detail: detailPart }));
    return { type: 'subpoints', property, subpoints };
  }
  return { type: 'simple', property, value: valueStr };
};

/** Renders operator spec lines with styled property and value; subpoints rendered as indented list. */
const specList = (lines) => (
  <ul className="operator-spec-list">
    {lines.filter(Boolean).map((line, i) => {
      const parsed = parseSpecLineWithSubpoints(line);
      if (parsed.type === 'subpoints') {
        return (
          <li key={i} className="operator-spec-item operator-spec-item--with-sublist">
            <span className="spec-property">{parsed.property}</span>
            <span className="spec-separator"> : </span>
            <ul className="operator-spec-sublist">
              {parsed.subpoints.map(({ option, detail }, j) => (
                <li key={j} className="operator-spec-subitem">
                  <span className="spec-subpoint-option">{option}</span>
                  {detail && (
                    <>
                      <span className="spec-subpoint-arrow"> → </span>
                      <span className="spec-value">{detail}</span>
                    </>
                  )}
                </li>
              ))}
            </ul>
          </li>
        );
      }
      const [property, value] = parseSpecLine(line);
      return (
        <li key={i} className="operator-spec-item">
          {property ? (
            <>
              <span className="spec-property">{property}</span>
              <span className="spec-separator"> : </span>
              <span className="spec-value">{value}</span>
            </>
          ) : (
            <span className="spec-value">{value}</span>
          )}
        </li>
      );
    })}
  </ul>
);

/** Per-operator content for the detail view (from Leaf_Content.txt). */
const OPERATOR_CONTENT = {
  Conv2D: specList([
    'Filter Size : No Hard Limitations',
    'Kernel Size : Symmetric & Asymmetric : Limitation K1 X K2 <= 400',
    'Strides : Symmetric & Asymmetric',
    'Activations : Linear & ReLU & Sigmoid & TanH & LeakyReLU [Alpha value defaults to 0.2] & Softmax',
    'Padding : Valid & Same',
    'Data Format : Channels Last & Channels First',
    'Dilation Rate : (1, 1)',
    'Groups : 1',
    'Use Bias : True & False',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.',
    'Input Shape : No Hard Limitations'
  ]),
  Conv1D: specList([
    'Filter Size : No Hard Limitations',
    'Kernel Size : Limitation K <= 400',
    'Strides : No Hard Limitations',
    'Activations : Linear & ReLU & Sigmoid & TanH & LeakyReLU [Alpha value defaults to 0.2] & Softmax',
    'Padding : Valid & Same',
    'Data Format : Channels Last & Channels First',
    'Dilation Rate : 1',
    'Groups : 1',
    'Use Bias : True & False',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.',
    'Input Shape : No Hard Limitations'
  ]),
  DepthwiseConv2D: specList([
    'Kernel Size : Symmetric & Asymmetric : Limitation K1 X K2 <= 400',
    'Strides : Symmetric & Asymmetric',
    'Activations : Linear & ReLU & Sigmoid & TanH & LeakyReLU [Alpha value defaults to 0.2] & Softmax',
    'Padding : Valid & Same',
    'Depth Multiplier : No Hard Limitations',
    'Data Format : Channels Last & Channels First',
    'Dilation Rate : (1, 1)',
    'Groups : 1',
    'Use Bias : True & False',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.',
    'Input Shape : No Hard Limitations'
  ]),
  DepthwiseConv1D: specList([
    'Kernel Size : Limitation K <= 400',
    'Strides : No Hard Limitations',
    'Activations : Linear & ReLU & Sigmoid & TanH & LeakyReLU [Alpha value defaults to 0.2] & Softmax',
    'Padding : Valid & Same',
    'Depth Multiplier : No Hard Limitations',
    'Data Format : Channels Last & Channels First',
    'Dilation Rate : 1',
    'Groups : 1',
    'Use Bias : True & False',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.',
    'Input Shape : No Hard Limitations'
  ]),
  SeparableConv2D: specList([
    'Filter Size : No Hard Limitations',
    'Kernel Size : Symmetric & Asymmetric : Limitation K1 X K2 <= 400',
    'Strides : Symmetric & Asymmetric',
    'Activations : Linear & ReLU & Sigmoid & TanH & LeakyReLU [Alpha value defaults to 0.2] & Softmax',
    'Padding : Valid & Same',
    'Depth Multiplier : No Hard Limitations',
    'Data Format : Channels Last & Channels First',
    'Dilation Rate : (1, 1)',
    'Groups : 1',
    'Use Bias : True & False',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.',
    'Input Shape : No Hard Limitations'
  ]),
  SeparableConv1D: specList([
    'Filter Size : No Hard Limitations',
    'Kernel Size : Limitation K <= 400',
    'Strides : No Hard Limitations',
    'Activations : Linear & ReLU & Sigmoid & TanH & LeakyReLU [Alpha value defaults to 0.2] & Softmax',
    'Padding : Valid & Same',
    'Depth Multiplier : No Hard Limitations',
    'Data Format : Channels Last & Channels First',
    'Dilation Rate : 1',
    'Groups : 1',
    'Use Bias : True & False',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.',
    'Input Shape : No Hard Limitations'
  ]),
  MaxPooling2D: specList([
    'Pool Size : Symmetric & Asymmetric : Limitation P1 X P2 <= 400',
    'Strides : Symmetric & Asymmetric',
    'Padding : Valid & Same',
    'Data Format : Channels Last & Channels First'
  ]),
  MaxPooling1D: specList([
    'Pool Size : Limitation P <= 400',
    'Strides : No Hard Limitations',
    'Padding : Valid & Same',
    'Data Format : Channels Last & Channels First'
  ]),
  AveragePooling2D: specList([
    'Pool Size : Symmetric & Asymmetric : Limitation P1 X P2 <= 400',
    'Strides : Symmetric & Asymmetric',
    'Padding : Valid & Same',
    'Data Format : Channels Last & Channels First'
  ]),
  AveragePooling1D: specList([
    'Pool Size : Limitation P <= 400',
    'Strides : No Hard Limitations',
    'Padding : Valid & Same',
    'Data Format : Channels Last & Channels First'
  ]),
  Dense: specList([
    'Units : No Hard Limitations',
    'Activations : Linear & ReLU & Sigmoid & TanH & LeakyReLU [Alpha value defaults to 0.2] & Softmax',
    'Use Bias : True & False',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.'
  ]),
  LSTM: specList([
    'Units : No Hard Limitations',
    'Activation : TanH & Sigmoid',
    'Recurrent Activation : Sigmoid & Hard Sigmoid & TanH',
    'Use Bias : True & False',
    'Unit Forget Bias : True',
    'Dropout : 0 to 1',
    'Recurrent Dropout : 0 to 1',
    'Return Sequences : True & False',
    'Return State : False',
    'Go Backwards : False',
    'Stateful : False',
    'Unroll : False',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.'
  ]),
  BatchNormalization: specList([
    'Axis : CNN 1D CL : {-1, 2} & CNN 1D CF : {1, -2} & CNN 2D CL : {-1, 3} & CNN 2D CF : {1, -3} & Dense : {1, -1} & LSTM Return Sequences True : {2, -1} & LSTM Return Sequences False : {1, -1}',
    'Trainable : True & False',
    'Momentum : No Limitation',
    'Epsilon : No Limitation',
    'Center : True',
    'Scale : True',
    'All Initialisers, Regularisers & Constraints for Layer Generation Supported.'
  ]),
  Dropout: specList([
    'Rate : 0 to 1',
    'Noise : No Limitation',
    'Seed : No Limitation'
  ]),
  SpatialDropout1D: specList([
    'Rate : 0 to 1',
    'Seed : No Limitation'
  ]),
  SpatialDropout2D: specList([
    'Rate : 0 to 1',
    'Seed : No Limitation',
    'Data Format : Channels Last & Channels First'
  ]),
  AlphaDropout: specList([
    'Rate : 0 to 1',
    'Noise : No Limitation',
    'Seed : No Limitation'
  ]),
  GaussianDropout: specList([
    'Rate : 0 to 1',
    'Seed : No Limitation'
  ]),
  GaussianNoise: specList([
    'Stddev : No Limitation',
    'Seed : No Limitation'
  ]),
  ActivityRegularization: specList([
    'L1 : No Limitation',
    'L2 : No Limitation'
  ]),
  Flatten: specList([
    'Data Format : Channels Last '
  ]),
  Reshape: specList([
    'Target Shape : No Hard Limitations'
  ]),
  Permute: specList([
    'Dims : No Hard Limitations'
  ]),
  Input: specList([
    'Shape : No Hard Limitations',
    'Batch Size : None',
    'Dtype : No Hard Limitations [Default float32]',
    'Sparse : False',
    'Ragged : False',
    'Batch Shape : None'
  ]),
  Activation: specList([
    'Activation : Linear & ReLU & Sigmoid & TanH & LeakyReLU [Alpha value defaults to 0.2] & Softmax'
  ]),
  ReLU: specList([
    'Max Value : None',
    'Negative_slope : 0.0',
    'Threshold : 0.0'
  ]),
  LeakyReLU: specList([
    'Alpha : Customizable [0.3 by Default]'
  ]),
  Softmax: specList([
    'Axis : No Limitation (Default -1)'
  ])
};

const SupportedOperators = () => {
  const [openSection, setOpenSection] = useState(null);
  const [showOperatorDetail, setShowOperatorDetail] = useState(false);
  const [selectedOperator, setSelectedOperator] = useState(null);
  const [operatorDetailExiting, setOperatorDetailExiting] = useState(false);
  const [heights, setHeights] = useState({
    convolution: '0px',
    pool: '0px',
    core: '0px',
    lstm: '0px',
    activation: '0px',
    normalization: '0px',
    regularization: '0px',
    input: '0px',
    shape: '0px',
    dropout: '0px'
  });

  const convolutionRef = useRef(null);
  const denseRef = useRef(null);
  const poolRef = useRef(null);
  const recurrentRef = useRef(null);
  const activationRef = useRef(null);
  const normalizationRef = useRef(null);
  const regularizationRef = useRef(null);
  const inputRef = useRef(null);
  const shapeRef = useRef(null);
  const dropoutRef = useRef(null);

  const handleOperatorClick = (operatorName) => {
    setSelectedOperator(operatorName);
    setShowOperatorDetail(true);
  };

  const handleOperatorBack = () => {
    setOperatorDetailExiting(true);
  };

  const handleOperatorDetailAnimationEnd = () => {
    if (operatorDetailExiting) {
      setShowOperatorDetail(false);
      setSelectedOperator(null);
      setOperatorDetailExiting(false);
    }
  };

  const getRef = (section) => {
    switch (section) {
      case 'convolution': return convolutionRef;
      case 'pool': return poolRef;
      case 'core': return denseRef;
      case 'lstm': return recurrentRef;
      case 'activation': return activationRef;
      case 'normalization': return normalizationRef;
      case 'regularization': return regularizationRef;
      case 'input': return inputRef;
      case 'shape': return shapeRef;
      case 'dropout': return dropoutRef;
      default: return null;
    }
  };

  const expandSection = (section) => {
    if (openSection) {
      setHeights(prev => ({ ...prev, [openSection]: '0px' }));
    }
    const ref = getRef(section);
    if (ref?.current) {
      const newHeight = `${ref.current.scrollHeight}px`;
      setHeights(prev => ({ ...prev, [section]: newHeight }));
    }
    setOpenSection(section);
  };

  const collapseSection = () => {
    if (openSection) {
      setHeights(prev => ({ ...prev, [openSection]: '0px' }));
      setOpenSection(null);
    }
  };

  const toggleSection = (section) => {
    if (openSection === section) {
      collapseSection();
    } else {
      expandSection(section);
    }
  };

  if (showOperatorDetail) {
    return (
      <div
        className={`operator-detail-view ${operatorDetailExiting ? 'operator-detail-exit' : 'operator-detail-enter'}`}
        onAnimationEnd={handleOperatorDetailAnimationEnd}
      >
        <div className="operator-detail-header">
          <ArrowLeft
            className="back-arrow"
            size={18}
            strokeWidth={3}
            onClick={handleOperatorBack}
            title="Go back"
          />
          <span className="operator-detail-header-label">Back to Supported Operators</span>
        </div>
        <h2 className="operator-detail-heading">{selectedOperator}</h2>
        <div className="operator-detail-content">
          {OPERATOR_CONTENT[selectedOperator] ?? (
            <p>Content for this operator is not defined yet. Add an entry for &quot;{selectedOperator}&quot; in OPERATOR_CONTENT.</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="supported-operators-list-wrap">
      <h2 className="supported-operators-heading-fixed">Supported Operators</h2>
      <div className="supported-operators-scroll-body">
      {/* Convolution Layers */}
      <div className={`section ${openSection === 'convolution' ? 'active' : ''}`} onMouseEnter={() => expandSection('convolution')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('convolution')}>
          <div className="icon-title">
            <img src="/assets/Convolution.png" alt="Convolution" className="section-icon" />
            <span>Convolution Layers</span>
          </div>
        </div>
        <div ref={convolutionRef} className={`collapse-content ${openSection === 'convolution' ? 'open' : ''}`} style={{ height: heights.convolution }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('Conv1D')}>Conv1D</button>
            <button type="button" onClick={() => handleOperatorClick('Conv2D')}>Conv2D</button>
            <button type="button" onClick={() => handleOperatorClick('DepthwiseConv1D')}>DepthwiseConv1D</button>
            <button type="button" onClick={() => handleOperatorClick('DepthwiseConv2D')}>DepthwiseConv2D</button>
            <button type="button" onClick={() => handleOperatorClick('SeparableConv1D')}>SeparableConv1D</button>
            <button type="button" onClick={() => handleOperatorClick('SeparableConv2D')}>SeparableConv2D</button>
          </div>
        </div>
      </div>

      {/* Pooling Layers */}
      <div className={`section ${openSection === 'pool' ? 'active' : ''}`} onMouseEnter={() => expandSection('pool')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('pool')}>
          <div className="icon-title">
            <img src="/assets/Pooling_img.png" alt="Pooling" className="section-icon" />
            <span>Pooling Layers</span>
          </div>
        </div>
        <div ref={poolRef} className={`collapse-content ${openSection === 'pool' ? 'open' : ''}`} style={{ height: heights.pool }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('MaxPooling1D')}>MaxPooling1D</button>
            <button type="button" onClick={() => handleOperatorClick('MaxPooling2D')}>MaxPooling2D</button>
            <button type="button" onClick={() => handleOperatorClick('AveragePooling1D')}>AveragePooling1D</button>
            <button type="button" onClick={() => handleOperatorClick('AveragePooling2D')}>AveragePooling2D</button>
          </div>
        </div>
      </div>

      {/* Core Layers */}
      <div className={`section ${openSection === 'core' ? 'active' : ''}`} onMouseEnter={() => expandSection('core')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('core')}>
          <div className="icon-title">
            <img src="/assets/dense.png" alt="Core" className="section-icon" />
            <span>Core Layers</span>
          </div>
        </div>
        <div ref={denseRef} className={`collapse-content ${openSection === 'core' ? 'open' : ''}`} style={{ height: heights.core }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('Dense')}>Dense</button>
          </div>
        </div>
      </div>

      {/* Recurrent Layers */}
      <div className={`section ${openSection === 'lstm' ? 'active' : ''}`} onMouseEnter={() => expandSection('lstm')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('lstm')}>
          <div className="icon-title">
            <img src="/assets/Lstm.png" alt="Recurrent" className="section-icon" />
            <span>Recurrent Layers</span>
          </div>
        </div>
        <div ref={recurrentRef} className={`collapse-content ${openSection === 'lstm' ? 'open' : ''}`} style={{ height: heights.lstm }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('LSTM')}>LSTM</button>
          </div>
        </div>
      </div>

      {/* Normalization Layers */}
      <div className={`section ${openSection === 'normalization' ? 'active' : ''}`} onMouseEnter={() => expandSection('normalization')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('normalization')}>
          <div className="icon-title">
            <img src="/assets/Normalization.png" alt="Normalization" className="section-icon" />
            <span>Normalization Layers</span>
          </div>
        </div>
        <div ref={normalizationRef} className={`collapse-content ${openSection === 'normalization' ? 'open' : ''}`} style={{ height: heights.normalization }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('BatchNormalization')}>BatchNormalization</button>
          </div>
        </div>
      </div>

      {/* Regularization Layers */}
      <div className={`section ${openSection === 'regularization' ? 'active' : ''}`} onMouseEnter={() => expandSection('regularization')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('regularization')}>
          <div className="icon-title">
            <img src="/assets/regularization.png" alt="Regularization" className="section-icon" />
            <span>Regularization Layers</span>
          </div>
        </div>
        <div ref={regularizationRef} className={`collapse-content ${openSection === 'regularization' ? 'open' : ''}`} style={{ height: heights.regularization }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('GaussianNoise')}>GaussianNoise</button>
            <button type="button" onClick={() => handleOperatorClick('ActivityRegularization')}>ActivityRegularization</button>
          </div>
        </div>
      </div>

      {/* Input Layers */}
      <div className={`section ${openSection === 'input' ? 'active' : ''}`} onMouseEnter={() => expandSection('input')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('input')}>
          <div className="icon-title">
            <img src="/assets/Input_img.png" alt="Input" className="section-icon" />
            <span>Input Layers</span>
          </div>
        </div>
        <div ref={inputRef} className={`collapse-content ${openSection === 'input' ? 'open' : ''}`} style={{ height: heights.input }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('Input')}>Input</button>
          </div>
        </div>
      </div>

      {/* Shape Layers */}
      <div className={`section ${openSection === 'shape' ? 'active' : ''}`} onMouseEnter={() => expandSection('shape')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('shape')}>
          <div className="icon-title">
            <img src="/assets/Shape_img.png" alt="Shape" className="section-icon" />
            <span>Shape Layers</span>
          </div>
        </div>
        <div ref={shapeRef} className={`collapse-content ${openSection === 'shape' ? 'open' : ''}`} style={{ height: heights.shape }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('Flatten')}>Flatten</button>
            <button type="button" onClick={() => handleOperatorClick('Reshape')}>Reshape</button>
            <button type="button" onClick={() => handleOperatorClick('Permute')}>Permute</button>
          </div>
        </div>
      </div>

      {/* Activation Layers */}
      <div className={`section ${openSection === 'activation' ? 'active' : ''}`} onMouseEnter={() => expandSection('activation')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('activation')}>
          <div className="icon-title">
            <img src="/assets/activation.png" alt="Activation" className="section-icon" />
            <span>Activation Layers</span>
          </div>
        </div>
        <div ref={activationRef} className={`collapse-content ${openSection === 'activation' ? 'open' : ''}`} style={{ height: heights.activation }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('Activation')}>Activation</button>
            <button type="button" onClick={() => handleOperatorClick('ReLU')}>ReLU</button>
            <button type="button" onClick={() => handleOperatorClick('LeakyReLU')}>LeakyReLU</button>
            <button type="button" onClick={() => handleOperatorClick('Softmax')}>Softmax</button>
          </div>
        </div>
      </div>

      {/* Dropout Layers */}
      <div className={`section ${openSection === 'dropout' ? 'active' : ''}`} onMouseEnter={() => expandSection('dropout')} onMouseLeave={collapseSection}>
        <div className="section-header" onClick={() => toggleSection('dropout')}>
          <div className="icon-title">
            <img src="/assets/Dropout_img.png" alt="Dropout" className="section-icon" />
            <span>Dropout Layers</span>
          </div>
        </div>
        <div ref={dropoutRef} className={`collapse-content ${openSection === 'dropout' ? 'open' : ''}`} style={{ height: heights.dropout }}>
          <div className="button-group">
            <button type="button" onClick={() => handleOperatorClick('Dropout')}>Dropout</button>
            <button type="button" onClick={() => handleOperatorClick('SpatialDropout1D')}>SpatialDropout1D</button>
            <button type="button" onClick={() => handleOperatorClick('SpatialDropout2D')}>SpatialDropout2D</button>
            <button type="button" onClick={() => handleOperatorClick('AlphaDropout')}>AlphaDropout</button>
            <button type="button" onClick={() => handleOperatorClick('GaussianDropout')}>GaussianDropout</button>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
};

export default SupportedOperators;
