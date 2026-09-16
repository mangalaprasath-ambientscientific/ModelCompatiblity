export const fetchPreprocessingSteps = async (isCustom = false) => {
    const endpoint = isCustom
        ? "http://127.0.0.1:9262/api/custom-preprocessing-steps"
        : "http://127.0.0.1:9262/api/preprocessing-steps";

    const res = await fetch(endpoint);
    const data = await res.json();
    return data.steps;
};
