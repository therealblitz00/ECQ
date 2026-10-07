# Problem Description & Specification: Similarity Detection for Fashion Retail Products (Parfois)

## Context & Background
- **Company:** Parfois (Portuguese fast fashion accessories brand).
- **Scale:** 1000+ stores across 74 countries, launching ~4,200 SKUs per season with an average annual growth of 28% (2021–2023).
- **Core Problem:** Traditional restocking and purchasing face severe inefficiencies (risk of overstock, stockouts, and poor store-level distribution) due to a lack of data-driven item similarity insights.

---

## Objective
Develop a robust **similarity detection system** for fashion retail products capable of identifying items that are visually and/or categorically similar.

---

## Goals & Business Impact
1. **Trend Identification:** Detect emerging style trends by clustering or matching similar product characteristics.
2. **Inventory Optimization:** Prevent overstocking of redundant items and mitigate stockouts by analyzing cross-product substitution.
3. **Sales & Experience Enhancement:** Drive sales conversion by offering better product alternatives to customers and enhancing recommendations.

---

## Data Inputs
The solution must leverage three main data pillars:
1. **Master Product Data:** 
   - Category, Material, Family, Color, Price, Exposition Date, Description, Fashion Type.
2. **Accumulated Sales Data:** 
   - Sales Quantity, Sales Amount.
3. **Product Images:** 
   - High-resolution visual imagery of items for computer vision/multimodal feature extraction.

---

## Constraints & Challenges
- **High Dimensionality & Volume:** Must handle thousands of fast-changing SKUs per season with continuous feature updates.
- **Multimodal Alignment:** Successfully bridge unstructured visual features (images) with structured tabular data (category, color, sales performance).
- **Interpretability for Business Users:** Black-box similarity scores are insufficient; the system must highlight *why* products are similar so commercial buyers can make informed purchasing and distribution decisions.

---

## Expected Output Format
For **each target product**, the system must output:
1. **Similar Items:** At least 4 top-ranked similar products (identified by SKU ID).
2. **Explainable Features:** The most relevant characteristics (visual and categorical) explaining *why* these products are matched, allowing business evaluators to validate the recommendation.

---

## Example Structure
- **Query Product:** `167718_BU`
- **Similar Products:** `169597_FU`, `169607_IG`, `169607_PE`, `117781_LB`
- **Evaluation Payload:** Key matching attributes (color palette, shape/form factor from images, material family).