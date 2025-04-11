import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# ========== CÀI ĐẶT GIAO DIỆN CHUNG ==========
st.set_page_config(page_title="Recommendation System", layout="wide")
st.title("Recommendation System: ")
st.markdown("<h1 style='color:black; font-size: 30px;'> <b>Thời Trang Nam.</b></h1>", unsafe_allow_html=True)

menu = ["Home", "Project Introduction", "Achievements", "Users"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.markdown("---")
st.sidebar.image("shoppee_menu.jpg", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("👩‍🏫 **Giảng viên:**\n\nCô: Khuất Thùy Phương")
st.sidebar.markdown("🎖️ **Thực hiện bởi:**")
st.sidebar.info("Dương Đại Dũng")
st.sidebar.markdown("📅 **Ngày báo cáo:** 13/04/2025")

# ========== HÀM HIỂN THỊ SẢN PHẨM ==========
def display_recommended_products(recommended_products, cols=3):
    for i in range(0, len(recommended_products), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:
                    if isinstance(product['image'], str) and product['image']:
                        try:
                            st.image(product['image'], caption=product['product_name'], use_container_width=True)
                        except Exception as e:
                            st.warning(f"Không thể hiển thị ảnh cho sản phẩm '{product['product_name']}': {e}")
                    st.markdown(f"**{product['product_name']}**")
                    exp = st.expander(" Mô tả")
                    truncated = ' '.join(product['Content'].split()[:30]) + "..."
                    exp.write(truncated)

# ========== TRANG CHỦ ==========
if choice == 'Home':  
    st.subheader("[Trang chủ](https://shopee.vn/)")
    st.image('shoppee4.jpg', use_container_width=True)
    st.write('ĐỒ ÁN TỐT NGHIỆP: DL07_DATN_K302 ')

# ========== GIỚI THIỆU DỰ ÁN ==========
elif choice == 'Project Introduction':    
    st.markdown("### [Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    try:
        st.image('shoppee1.jpg', use_container_width=True)
    except:
        st.warning("Không tìm thấy ảnh 'shoppee1.jpg'!")

    st.write("""
        - Trong bối cảnh thị trường thời trang nam trực tuyến trên Shopee ngày càng phát triển, việc giúp người dùng dễ dàng khám phá những sản phẩm phù hợp với sở thích và phong cách cá nhân trở nên vô cùng quan trọng. Việc xây dựng một hệ thống gợi ý thông minh, nhằm giải quyết bài toán về sự quá tải thông tin và nâng cao trải nghiệm mua sắm thời trang nam cho người dùng Shopee.
        - Sử dụng hai phương pháp:
            - **Collaborative Filtering:**
            - **Content-Based Filtering:**
    """)

# ========== THÀNH TỰU ==========
elif choice == 'Achievements': 
    st.write('Collaborative Filtering:')
    st.image('SVD.jpg', use_container_width=True)
    st.write('==> Chọn model SVD của Surprise là tối ưu nhất.')
    st.write('Content-Based Filtering:')
    st.image('gensim_cosine.jpg', use_container_width=True)
    st.write("""
        - Xin phép sử dụng cosine similarity.
        - Xây hệ thống gợi ý tất cả sản phẩm --> nhiều người dùng: gensim (TfidfModel, Similarity)
    """)

# ========== NGƯỜI DÙNG ==========
elif choice == 'Users':
    st.image('shoppee2.jpg', use_container_width=True)

    # ===== LOAD DỮ LIỆU CẦN THIẾT =====
    with open('products_df.pkl', 'rb') as f:
        df_products = pickle.load(f)
    with open('recommendations_dict.pkl', 'rb') as f:
        recommendations_dict = pickle.load(f)
    with open("svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    with open("sample_df.pkl", "rb") as f:
        ratings_df = pickle.load(f)
    with open("products_name_df.pkl", "rb") as f:
        products_name_df = pickle.load(f)

    # ===== GỢI Ý CONTENT-BASED =====
    st.markdown("##  Gợi ý theo sản phẩm bạn đã chọn")

    #  Cố định random bằng random_state
    random_products = df_products.sample(n=20, random_state=42)

    # Tạo danh sách tùy chọn: [(Tên, ID)]
    product_options = [(row['product_name'], row['product_id']) for _, row in random_products.iterrows()]

    # Hiển thị dropdown để chọn sản phẩm
    selected_product = st.selectbox("Chọn sản phẩm", options=product_options, format_func=lambda x: x[0])

    if selected_product:
        selected_name = selected_product[0]
        selected_ma_sp = selected_product[1]

        st.write(" Tên sản phẩm đã chọn:", selected_name)
        st.write(" Mã sản phẩm:", selected_ma_sp)

        # Thông tin sản phẩm được chọn
        selected_info = df_products[df_products['product_id'] == selected_ma_sp]

        if not selected_info.empty:
            product_name = selected_info['product_name'].values[0]
            st.markdown(f"### {product_name}")

            # Mô tả ngắn gọn
            truncated_desc = ' '.join(selected_info['Content'].values[0].split()[:50]) + "..."
            st.write("**Mô tả:**", truncated_desc)

            # Gợi ý sản phẩm tương tự
            st.markdown("### Sản phẩm liên quan bạn có thể thích:")
            related_ids = recommendations_dict.get(selected_ma_sp, [])
            related_products = df_products[df_products['product_id'].isin(related_ids)]

            # Hiển thị gợi ý
            display_recommended_products(related_products, cols=3)

    # Ảnh cuối trang
    st.image('shoppee3.jpg', use_container_width=True)






    # ===== GỢI Ý THEO USER =====
    st.markdown("---")
    st.markdown("##  Gợi ý sản phẩm dựa trên hành vi người dùng (Hội viên)")
    st.write('Một số user_id ví dụ: 378643, 244923, 307455, 354, 97688.')
    user_input = st.text_input("Nhập mã người dùng (user_id):")

    if user_input:
        try:
            user_id = int(user_input)
            rated_products = ratings_df[ratings_df["user_id"] == user_id]["product_id"].tolist()
            all_products = df_products["product_id"].tolist()
            unrated_products = [pid for pid in all_products if pid not in rated_products]

            predictions = []
            for pid in unrated_products:
                pred = svd_model.predict(user_id, pid)
                predictions.append((pid, pred.est))

            top_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:9]
            top_product_ids = [pid for pid, _ in top_preds]
            recommended_df = df_products[df_products["product_id"].isin(top_product_ids)]

            st.markdown("###  Gợi ý theo hành vi người dùng:")
            display_recommended_products(recommended_df, cols=3)

        except ValueError:
            st.warning("Vui lòng nhập `user_id` hợp lệ (số nguyên).")






