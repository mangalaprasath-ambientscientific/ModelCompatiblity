import React, { useState, useEffect } from 'react';
import { FaCog, FaCheck, FaTimes, FaExclamationTriangle, FaChevronUp } from 'react-icons/fa';
import '../styles/modelValidation.css';

/* ----------------------------------------------------------------------------
 * Presentational only.  There is no backend any more: ModelCompatibility reads
 * the .h5 with jsfive, runs checkModelCompatibility() in the browser and hands
 * the result down as `results`.  This component never fetches.
 *
 * The contract is unchanged from what POST /validate-model used to return:
 *
 *   layer_info    [issueCount, totalLayers]   or [null, null] on a failed load
 *   layer_details [["Conv2D Supported.", "Operators Supported."], ...] or null
 *   tips          ["No Suggestions to Give."] or one line per problem
 *
 * Note the shape of a layer_details row.  Model_Compatibility.py builds it as
 *
 *     filtered_op[i][0] = filtered_op[i][0][0] + ' ' + filtered_op[i][0][1] + '.'
 *     filtered_op[i][1] = 'Operators Supported.' / 'Operators Not Supported.'
 *
 * so it is TWO cells - "<LayerType> <status>." and "Operators <status>." - not
 * four.  Reading layer[2] / layer[3] leaves both statuses undefined, which is
 * why every row used to render as Supported.
 * -------------------------------------------------------------------------- */
const ModelValidation = ({ modelFile, results, isValidating, onValidate }) => {
    const [showSuggestions, setShowSuggestions] = useState(false);

    /* A failed load has no layer table to look at, so open the reasons straight
       away; anything else waits for the user to ask. */
    useEffect(() => {
        setShowSuggestions(!!(results && !results.layer_details));
    }, [results]);

    const isModelCompatible = !!(
        results &&
        results.tips &&
        results.tips.length === 1 &&
        results.tips[0] === 'No Suggestions to Give.'
    );

    const layerDetailsFormatted = !results || !results.layer_details
        ? []
        : results.layer_details.map((layer) => {
            const head = String(layer[0] || '');      // "Conv2D Supported."
            const opsCell = String(layer[1] || '');   // "Operators Not Supported."
            const sp = head.indexOf(' ');
            return {
                layer_name: sp === -1 ? head : head.slice(0, sp),
                layer_supported: head.indexOf('Not Supported') === -1,
                operator_supported: opsCell.indexOf('Not Supported') === -1,
            };
        });

    const backendIssueCount = (results && results.layer_info && results.layer_info[0]) || 0;
    const backendTotalLayers = (results && results.layer_info && results.layer_info[1]) || 0;

    const totalIssues = results && results.tips && results.tips[0] !== 'No Suggestions to Give.'
        ? results.tips.length
        : 0;

    const hasSuggestions = !isModelCompatible && !!(results && results.tips && results.tips.length > 0);

    return (
        <>
            <div className="validation-container">
                {isValidating ? (
                    <div className="validation-loading">
                        <FaCog className="fa-spin" />
                        <p>Validating model...</p>
                    </div>
                ) : results ? (
                    <div className="validation-results">
                        <div className="validation-header">
                            <h3>Model Compatibility Results</h3>
                        </div>

                        {/* Compatibility Summary */}
                        <div className={`compatibility-summary ${isModelCompatible ? 'compatible' : 'incompatible'}`}>
                            <div className="summary-icon">
                                {isModelCompatible ? (
                                    <FaCheck className="success-icon" />
                                ) : (
                                    <FaTimes className="error-icon" />
                                )}
                            </div>
                            <div className="summary-content">
                                <h4>
                                    {isModelCompatible
                                        ? 'Model Layers are compatible with GPX.'
                                        : 'Model Layers are not compatible with GPX.'}
                                </h4>
                                <p>
                                    {isModelCompatible
                                        ? 'All layers and operators are compatible.'
                                        : `Issues found in ${backendIssueCount} out of ${backendTotalLayers} layers.`}
                                </p>
                            </div>
                            <div className="summary-icon" style={{ visibility: 'hidden' }}>
                                <FaCheck className="success-icon" />
                            </div>
                        </div>

                        {/* Layer Details Section */}
                        <div className="layer-details-section">
                            <div className="section-header">
                                <h4>Model Layer Details</h4>
                            </div>

                            <div className="layer-details-container expanded">
                                <div className="layer-details-header">
                                    <div className="header-name">Layer Name</div>
                                    <div className="header-status">Layer Status</div>
                                    <div className="header-status">Operator Status</div>
                                </div>
                                <div className="layers-list-container">
                                    {layerDetailsFormatted.map((layer, index) => (
                                        <div
                                            key={index}
                                            className={`layer-item ${layer.layer_supported ? 'supported' : 'unsupported'}`}
                                            style={{ animationDelay: `${index * 0.05}s` }}
                                        >
                                            <div className="layer-name">{layer.layer_name}</div>

                                            <div className="layer-status">
                                                {layer.layer_supported ? (
                                                    <span className="supported-badge">Supported</span>
                                                ) : (
                                                    <span className="unsupported-badge">Not supported</span>
                                                )}
                                            </div>
                                            <div className="layer-status">
                                                {layer.operator_supported ? (
                                                    <span className="supported-badge">Supported</span>
                                                ) : (
                                                    <span className="unsupported-badge">Not supported</span>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Suggestions - only when there is something to say */}
                        {showSuggestions && hasSuggestions && (
                            <div className="suggestions-box">
                                <div className="suggestions-header">
                                    <FaExclamationTriangle className="warning-icon" />
                                    <h4>Suggestions for Improvement</h4>
                                    <span className="count-badge">
                                        {totalIssues} {totalIssues === 1 ? 'issue to resolve' : 'issues to resolve'}
                                    </span>
                                </div>
                                <div className="suggestions-divider"></div>
                                <ul className="suggestions-list">
                                    {results.tips.map((tip, index) => (
                                        <li key={index} className="suggestion-item">
                                            {tip}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="validation-prompt">
                        <FaCheck className="box-icon" />
                        <h3>Validate Model</h3>
                        <p>Check GPX-10 compatibility and requirements</p>
                        <button
                            className="validate-btn"
                            onClick={onValidate}
                            disabled={!modelFile || !onValidate}
                        >
                            Run Validation
                        </button>
                    </div>
                )}
            </div>

            {hasSuggestions && !showSuggestions && (
                <div className="bottom-fade-container">
                    <div className="bottom-fade-overlay"></div>
                    <button
                        className="see-more-button"
                        onClick={() => setShowSuggestions(true)}
                    >
                        <FaChevronUp />
                        See more info
                    </button>
                </div>
            )}
        </>
    );
};

export default ModelValidation;
