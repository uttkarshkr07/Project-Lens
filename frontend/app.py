import io
import os

import requests
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Visual Electronics Search",
    page_icon="🔍",
    layout="wide",
)


def get_catalog_stats() -> dict | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/catalog/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def search_products(image_bytes: bytes) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/search",
        files={"image": ("query.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def load_image_from_url(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    except Exception:
        return None


def confidence_color(confidence: float) -> str:
    if confidence > 0.7:
        return "🟢"
    elif confidence >= 0.4:
        return "🟠"
    else:
        return "🔴"


# --- Sidebar ---
with st.sidebar:
    st.title("Visual Electronics Search")
    st.caption("Upload a photo of any electronics product — we'll identify it and find similar items.")
    st.markdown("---")
    st.markdown(f"[API Docs](http://localhost:8000/docs)")

    stats = get_catalog_stats()
    if stats:
        st.metric("Catalog Products", f"{stats.get('total_products', 0):,}")
    else:
        st.warning("Could not reach backend")

    with st.expander("How it works"):
        st.markdown(
            """
1. **YOLO** detects and crops the electronics product from your photo
2. **CLIP** (ViT-B/32) converts the cropped image into a 512-dim embedding
3. **Qdrant** searches the vector database for the most similar catalog items
4. **Gemini 2.0 Flash** analyzes the matches and generates a recommendation
"""
        )

# --- Main layout ---
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image_bytes = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Uploaded image", use_container_width=True)

    search_clicked = st.button("Search", type="primary", disabled=uploaded_file is None)

with right_col:
    if search_clicked and image_bytes is not None:
        with st.spinner("Detecting product and searching catalog..."):
            try:
                result = search_products(image_bytes)
            except requests.exceptions.ConnectionError:
                st.error("Backend service unavailable. Is Docker Compose running?")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"Search failed: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        detected_category = result.get("detected_category", "unknown")
        detected_confidence = result.get("detected_confidence", 0.0)
        matches = result.get("matches", [])
        recommendation = result.get("recommendation", "")

        # Detection badge
        color = confidence_color(detected_confidence)
        st.markdown(
            f"**Detected:** {color} `{detected_category}` &nbsp;|&nbsp; "
            f"**Confidence:** `{detected_confidence:.0%}`"
        )
        st.markdown("---")

        # Gemini recommendation
        if recommendation and "Recommendation unavailable" not in recommendation:
            with st.container():
                st.markdown("### ✨ AI Recommendation")
                st.markdown(recommendation)
        elif recommendation:
            st.warning("AI recommendation unavailable. Showing visual matches only.")

        st.markdown("---")

        # Product matches
        if not matches:
            st.warning("No similar products found. Try a clearer image.")
        else:
            st.subheader(f"Top {len(matches)} Visual Matches")
            for rank, match in enumerate(matches, 1):
                with st.container():
                    st.markdown(f"**#{rank}**")
                    img_col, info_col = st.columns([1, 3])

                    with img_col:
                        product_img = load_image_from_url(match.get("image_url", ""))
                        if product_img:
                            st.image(product_img, use_container_width=True)
                        else:
                            st.markdown("_(No image)_")

                    with info_col:
                        price = match.get("price", 0)
                        rating = match.get("rating", 0)
                        score = match.get("score", 0)
                        specs = match.get("specs", {})
                        top_specs = list(specs.values())[:3]

                        st.markdown(f"**{match.get('name', 'N/A')}**")
                        st.markdown(
                            f"Brand: `{match.get('brand', 'N/A')}` &nbsp;|&nbsp; "
                            f"Category: `{match.get('category', 'N/A')}`"
                        )
                        st.markdown(f"Price: **₹{price:,.0f}** &nbsp;|&nbsp; Rating: ⭐ {rating}")
                        st.progress(float(score), text=f"Similarity: {score:.1%}")
                        if top_specs:
                            st.markdown("\n".join(f"- {s}" for s in top_specs))

                    st.markdown("---")
    elif not search_clicked:
        st.info("Upload an image and click **Search** to find similar electronics products.")
