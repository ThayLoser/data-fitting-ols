import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from scipy import stats

class DataPipeline:
    """
    Class DataPipeline đóng gói toàn bộ quy trình tiền xử lý dữ liệu (Preprocessing)
    bao gồm Điền khuyết, Cắt râu (Winsorize), Biến đổi Box-Cox, Gộp nhóm và One-Hot Encoding.
    Đảm bảo triệt tiêu hoàn toàn rủi ro Data Leakage và khớp 100% với xử lý thủ công.
    """
    def __init__(self, n_neighbors=5, winsorize_lower=0.01, winsorize_upper=0.99, test_size=0.2, random_state=42):
        self.n_neighbors = n_neighbors
        self.winsorize_lower = winsorize_lower
        self.winsorize_upper = winsorize_upper
        self.test_size = test_size
        self.random_state = random_state
        
        # Các biến trạng thái "trí nhớ" để học từ tập Train
        self.modes_ = {}
        self.medians_ = {}
        self.knn_ = None
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.lambdas_ = {}
        self.encoded_columns_ = None
        self.num_cols_ = None
        self.cat_cols_ = None
        
    def fit(self, X, y=None):
        """Học (Fit) các tham số thống kê từ tập dữ liệu huấn luyện."""
        df = X.copy()
        
        # 1. Nhận diện cột số và cột chữ
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # 2. Xử lý content_rating đặc biệt
        if 'content_rating' in df.columns:
            df['content_rating'] = df['content_rating'].fillna('Unrated')
            
        # 3. Học Mode cho dữ liệu phân loại (trừ content_rating)
        for col in self.cat_cols_:
            if col != 'content_rating':
                self.modes_[col] = df[col].mode()[0]
                df[col] = df[col].fillna(self.modes_[col])
                
        # 4. Học Median cho dữ liệu số (trừ budget)
        for col in self.num_cols_:
            if col != 'budget':
                self.medians_[col] = df[col].median()
                df[col] = df[col].fillna(self.medians_[col])
                
        # 5. Học KNN Imputer riêng cho Budget
        if 'budget' in self.num_cols_:
            self.knn_ = KNNImputer(n_neighbors=self.n_neighbors)
            df[['budget']] = self.knn_.fit_transform(df[['budget']])
            
        # 6. Học phân vị để Cắt râu (Winsorize)
        for col in self.num_cols_:
            self.lower_bounds_[col] = df[col].quantile(self.winsorize_lower)
            self.upper_bounds_[col] = df[col].quantile(self.winsorize_upper)
            df[col] = df[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
            
        # 7. Học hệ số Lambda cho biến đổi Box-Cox
        skew_cols = ['duration', 'director_facebook_likes', 'actor_3_facebook_likes', 
                     'actor_1_facebook_likes', 'budget', 'actor_2_facebook_likes']
        for col in skew_cols:
            if col in df.columns:
                train_data = df[col].to_numpy() + 1e-6
                _, lmbda = stats.boxcox(train_data)
                self.lambdas_[col] = lmbda
                df[col] = stats.boxcox(train_data, lmbda=lmbda)
                
        # 8. Gom nhóm biến phân loại theo logic đã xác định
        df = self._group_categories(df)
        
        # 9. Học cấu trúc cột One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=self.cat_cols_, drop_first=True, dtype=float)
        self.encoded_columns_ = df_encoded.columns.tolist()
        
        return self
        
    def transform(self, X):
        """Áp dụng (Transform) các tham số đã học lên dữ liệu mới."""
        df = X.copy()
        
        # 1. Điền khuyết content_rating
        if 'content_rating' in df.columns:
            df['content_rating'] = df['content_rating'].fillna('Unrated')
            
        # 2. Điền khuyết Mode
        for col in self.cat_cols_:
            if col in df.columns and col != 'content_rating':
                df[col] = df[col].fillna(self.modes_[col])
                
        # 3. Điền khuyết Median
        for col in self.num_cols_:
            if col in df.columns and col != 'budget':
                df[col] = df[col].fillna(self.medians_[col])
                
        # 4. Điền khuyết KNN riêng cho Budget
        if 'budget' in self.num_cols_ and 'budget' in df.columns:
            df[['budget']] = self.knn_.transform(df[['budget']])
            
        # 5. Cắt râu Winsorize
        for col in self.num_cols_:
            if col in df.columns:
                df[col] = df[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
                
        # 6. Biến đổi Box-Cox
        for col, lmbda in self.lambdas_.items():
            if col in df.columns:
                test_data = df[col].to_numpy() + 1e-6
                df[col] = stats.boxcox(test_data, lmbda=lmbda)
                
        # 7. Gom nhóm biến phân loại
        df = self._group_categories(df)
        
        # 8. Mã hóa One-hot
        df_encoded = pd.get_dummies(df, columns=self.cat_cols_, drop_first=True, dtype=float)
        
        # 9. Đồng bộ hóa (Align) cấu trúc cột
        for col in self.encoded_columns_:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0
                
        df_encoded = df_encoded[self.encoded_columns_]
        
        return df_encoded
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
        
    def _group_categories(self, df):
        """Hàm phụ trợ: Gom các nhãn hiếm thành 'Other'."""
        if 'language' in df.columns:
            df['language'] = df['language'].apply(lambda x: x if x == 'English' else 'Other')
        if 'country' in df.columns:
            df['country'] = df['country'].apply(lambda x: x if x in ['USA', 'UK'] else 'Other')
        if 'content_rating' in df.columns:
            df['content_rating'] = df['content_rating'].apply(lambda x: x if x in ['R', 'PG-13', 'PG'] else 'Other')
        return df

    def process(self, raw_dataset):
        from sklearn.model_selection import train_test_split
        
        df = raw_dataset.copy()
        
        # 1. Loại bỏ các cột đa cộng tuyến, kém tương quan, rò rỉ dữ liệu và cardinality cao
        cols_to_drop = [
            'cast_total_facebook_likes', 'num_user_for_reviews',
            'facenumber_in_poster', 'aspect_ratio',
            'gross', 'num_voted_users', 'num_critic_for_reviews', 'movie_facebook_likes',
            'director_name', 'genres', 'actor_1_name', 'actor_2_name', 
            'actor_3_name', 'plot_keywords', 'movie_imdb_link'
        ]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Đặt tên phim làm index
        if 'movie_title' in df.columns:
            df['movie_title'] = df['movie_title'].str.strip()
            df = df.set_index('movie_title')
            
        # 2. Train-Test Split (Chưa xóa trùng lặp, chia xong mới xóa giống hệt notebook)
        train_set, test_set = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        
        # 3. Loại bỏ trùng lặp độc lập trên 2 tập
        train_set = train_set.drop_duplicates(keep='first')
        test_set = test_set.drop_duplicates(keep='first')
        
        # 4. Xóa dòng thiếu mục tiêu (imdb_score và title_year)
        train_set = train_set.dropna(subset=['imdb_score', 'title_year'])
        test_set = test_set.dropna(subset=['imdb_score', 'title_year'])
        
        # 5. Tách X, y
        y_train = train_set['imdb_score'].to_numpy()
        X_train_raw = train_set.drop(columns=['imdb_score'])
        
        y_test = test_set['imdb_score'].to_numpy()
        X_test_raw = test_set.drop(columns=['imdb_score'])
        
        # 6. Pipeline chạy
        X_train_clean_df = self.fit_transform(X_train_raw)
        X_test_clean_df = self.transform(X_test_raw)
        
        # 7. Trả về NumPy arrays
        X_train = X_train_clean_df.to_numpy()
        X_test = X_test_clean_df.to_numpy()
        
        return X_train, y_train, X_test, y_test
