#Install streamlit option menu dengan code di samping (pip install streamlit-option-menu)
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Klasifikasi Stunting Pada Balita",
    page_icon='https://i.imgur.com/Fe489Ox.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""<h1>Aplikasi Klasifikasi Stunting Pada Balita</h1>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h2 style = "text-align: center;"><img src="https://i.imgur.com/Fe489Ox.png" width="160" height="160"><br></h2>""",unsafe_allow_html=True), 
        ["Home", "Data", "Modeling", "Implementation"], 
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#FFC443"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#FFC443"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://i0.wp.com/deddyhuang.com/wp-content/uploads/2022/07/Cegah-Stunting.png?fit=811%2C400&ssl=1" width="700" height="400">
        </h3>""",unsafe_allow_html=True)

    elif selected == "Data":
        st.subheader("""Deskripsi Aplikasi""")
        st.write("""
         Aplikasi Ini digunakan untuk proses penentuan keputusan dan kelasifikasi penderita stunting pada balita berdasarkan indikator yang tersedia pada data balita stunting Puskesmas Kalianget Kabupaten Sumenep. 
        """)

        st.subheader("""Deskripsi Data""")
        st.write("""
        Data yang digunakan dalam aplikasi ini yaitu data balita penderita Stunting Puskesmas Kalianget Kabupaten Sumenep periode 2022-2023. Adapun beberapa indikator yang terdapat pada data tersebut yaitu:            
        """)
        st.write("""
        <ol>
            <li>Nama Lengkap Balita</li>
            <li>Jenis Kelamin Balita (L/P)</li>
            <li>Tanggal Lahir</li>
            <li>Umur Anak (Bulan)</li>
            <li>BB Lahir (Kg)</li>
            <li>PB Lahir (cm)</li>
            <li>BB Saat Ini (Kg)</li>
            <li>PB Saat Ini (cm)</li>
            <li>Nama Orang Tua</li>
            <li>Desa Domisili</li>
            <li>Status Gizi (TB/U)</li>
            <li>Status Stunting</li>
        </ol>
        """,unsafe_allow_html=True) 

        st.subheader("""Dataset Balita Penderita Stunting Puskesmas Kalianget Kabupaten Sumenep""")
        df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/Data_Kallianget_Semua_Fitur_Sistem.csv')
        st.dataframe(df.style.format({'BB Lahir (kg)': '{:.2f}','BB saat ini (kg)': '{:.2f}', 'TB saat ini (cm)': '{:.2f}'}))

    elif selected == "Modeling":
        with st.form("Modeling"):
            population = st.number_input('Masukkan Nilai Population Size', format='%.0f')
            generation = st.number_input('Masukkan Nilai Generation', format='%.0f')
            crossover = st.number_input('Masukkan Nilai Crossover Rate (0.00 - 1.00)', min_value=0.00, max_value=1.00)
            mutation = st.number_input('Masukkan Nilai Mutation Rate (0.00 - 1.00)', min_value=0.00, max_value=1.00)
            submit = st.form_submit_button("Submit")
            if submit:
                def modelingGASVM(population, generation, crossover, mutation, data):
                    X = data.drop(columns=["Status Stunting"])
                    y = data["Status Stunting"]

                    # Normalisasi
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(X)
                    features_names = X.columns.copy()
                    scaled_features = pd.DataFrame(scaled, columns=features_names)

                    # Pisahkan data menjadi data latih dan data uji
                    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

                    # Define the DEAP creator for the fitness function and individuals
                    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

                    # Fungsi untuk inisialisasi populasi awal
                    def initialize_population(pop_size, num_features):
                        return [creator.Individual(np.random.randint(2, size=num_features)) for _ in range(pop_size)]

                    # Fungsi SVM menggunakan scikit-learn
                    def sklearn_svm(X_train, y_train, X_test, y_test, selected_features):
                        X_train_selected = X_train.iloc[:, selected_features].copy().to_numpy()
                        X_test_selected = X_test.iloc[:, selected_features].copy().to_numpy()


                        clf = SVC(gamma=1.0, C=4.0, kernel="rbf", max_iter=10)  # Menggunakan kernel linear sebagai contoh
                        clf.fit(X_train_selected, y_train)
                        y_pred = clf.predict(X_test_selected)

                        accuracy = accuracy_score(y_test, y_pred)
                        return accuracy, y_pred

                    # Fungsi untuk menghitung nilai fitness dari setiap individu dalam populasi
                    def calculate_fitness(individual, X_train, y_train, X_test, y_test):
                        selected_features = np.where(individual == 1)[0]

                        if len(selected_features) == 0:
                            return (0,)
                        else:
                            accuracy, _ = sklearn_svm(X_train, y_train, X_test, y_test, selected_features)
                            return (accuracy,)

                    # Fungsi utama untuk algoritma genetika
                    def genetic_algorithm(X_train, X_test, y_train, y_test, pop_size, num_generations, cxpb, mutpb):
                        num_features = X_train.shape[1]

                        # Inisialisasi populasi awal
                        population = initialize_population(pop_size, num_features)

                        # Register DEAP operators
                        toolbox = base.Toolbox()
                        toolbox.register("evaluate", calculate_fitness, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                        toolbox.register("mate", tools.cxTwoPoint)
                        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
                        toolbox.register("select", tools.selTournament, tournsize=3)

                        # Evaluate the entire population
                        fitnesses = list(map(toolbox.evaluate, population))
                        for ind, fit in zip(population, fitnesses):
                            ind.fitness.values = fit

                        result_data = []
                        for generation in range(num_generations):
                            # Select the next generation individuals
                            offspring = toolbox.select(population, len(population))

                            # Clone the selected individuals
                            offspring = list(map(toolbox.clone, offspring))

                            # Apply crossover and mutation on the offspring
                            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                                if np.random.rand() < cxpb:
                                    toolbox.mate(child1, child2)
                                    del child1.fitness.values
                                    del child2.fitness.values

                            for mutant in offspring:
                                if np.random.rand() < mutpb:
                                    toolbox.mutate(mutant)
                                    del mutant.fitness.values

                            # Evaluate the individuals with an invalid fitness
                            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                            fitnesses = map(toolbox.evaluate, invalid_ind)
                            for ind, fit in zip(invalid_ind, fitnesses):
                                ind.fitness.values = fit

                            # Replace the old population by the offspring
                            population[:] = offspring

                            # Gather all the fitnesses in one list and print the statistics
                            fits = [ind.fitness.values[0] for ind in population]
                            length = len(population)
                            mean = sum(fits) / length
                            sum2 = sum(x*x for x in fits)
                            std = np.sqrt(sum2 / length - mean**2)

                            best_individual = tools.selBest(population, 1)[0]
                            fitur = [index for index, value in enumerate(best_individual) if value]

                            result_data.append({
                                'Generation': generation + 1,
                                'Population Size': pop_size,
                                'Crossover Rate': cxpb,
                                'Mutation Rate': mutpb,
                                'Fitness': max(fits),
                                'Avg Fitness': mean,
                                'Best Individual': best_individual,
                                'Selected Feature':fitur
                            })

                            result_df = pd.DataFrame(result_data)
                        return best_individual, result_df

                    pop_size = int(population)  # Ukuran populasi
                    num_generations = int(generation)  # Jumlah generasi
                    cxpb = crossover  # Probabilitas crossover
                    mutpb = mutation  # Probabilitas mutasi

                    best_individual,result_df = genetic_algorithm(X_train, X_test, y_train, y_test, pop_size, num_generations, cxpb, mutpb)

                    # Output fitur terbaik yang dipilih
                    selected_features = np.where(best_individual == 1)[0]
                    selected_feature_names = [features_names[i] for i in selected_features]

                    # Evaluasi model dengan SVM menggunakan fitur terbaik
                    X_train_selected = X_train.iloc[:, selected_features].copy().to_numpy()
                    X_test_selected = X_test.iloc[:, selected_features].copy().to_numpy()

                    clf = SVC(gamma=1.0, C=4.0, kernel="rbf", max_iter=10)   # Menggunakan kernel linear sebagai contoh
                    clf.fit(X_train_selected, y_train)
                    y_pred = clf.predict(X_test_selected)

                    precision = precision_score(y_test, y_pred)*100
                    recall = recall_score(y_test, y_pred)*100
                    f1 = f1_score(y_test, y_pred)*100
                    accuracy = accuracy_score(y_test, y_pred)*100

                    cm = confusion_matrix(y_test, y_pred)

                    return(scaled_features, result_df, selected_feature_names,precision,recall,f1,accuracy,cm)
                
                ga_input = pd.DataFrame((
                    [["Population Size",'%.0f' % population],["Generation",'%.0f' % generation],["Crossover Rate",'%.2f' % crossover],["Mutation Rate",'%.2f' % mutation]]
                ))

                st.header("Dataset Stunting")
                dataframe = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/Data_Kallianget_Semua_Fitur_Sistem.csv')
                st.dataframe(dataframe)

                st.header("Dataset Stunting untuk Klasifikasi")
                df_sistem = dataframe.drop(columns=["Nama Lengkap","Tanggal Lahir (DD-MM-YY)","Nama Ortu","Desa Domisili"])

                label_encoder = LabelEncoder()
                status_gizi_encode = label_encoder.fit_transform(df_sistem['Status Gizi (TB/U)'])+ 1 
                status_gizi_encode_df = pd.DataFrame(status_gizi_encode,columns = ['Status Gizi (TB/U)'])
                jenis_kelamin_encode = label_encoder.fit_transform(df_sistem['Jenis Kelamin (L/P)'])
                jenis_kelamin_encode_flipped = np.abs(jenis_kelamin_encode - 1)
                jenis_kelamin_encode_df = pd.DataFrame(jenis_kelamin_encode_flipped,columns = ['Jenis Kelamin (L/P)'])
                status_stunting_encode = label_encoder.fit_transform(df_sistem['Status Stunting'])
                status_stunting_encode_new = 1 - 2 * status_stunting_encode
                status_stunting_encode_new = pd.DataFrame(status_stunting_encode_new,columns = ['Status Stunting'])

                df_sistem_new = pd.concat([jenis_kelamin_encode_df,df_sistem.drop(columns=['Jenis Kelamin (L/P)','Status Gizi (TB/U)','Status Stunting']),status_gizi_encode_df,status_stunting_encode_new],axis=1)

                st.dataframe(df_sistem_new.style.format({'BB Lahir (kg)':'{:.2f}','BB saat ini (kg)': '{:.2f}', 'TB saat ini (cm)': '{:.2f}'}))
                
                scaled_features, result_df, selected_feature_names,precision,recall,f1,accuracy,cm = modelingGASVM(population,generation,crossover,mutation,df_sistem_new)

                st.header("Hasil Normaliasasi data")
                st.dataframe(scaled_features)

                st.header("Parameter Input Genetika Algoritma")
                st.dataframe(ga_input)

                st.header("Hasil Evaluasi Seleksi Fitur GA")
                st.dataframe(result_df)

                st.header("Grafik Kinerja GA-SVM")
                st.bar_chart(data=result_df, x="Generation", y=["Fitness","Avg Fitness"], width=0, height=0, use_container_width=True)

                st.header("Fitur Terseleksi")
                st.dataframe(selected_feature_names)

                st.header("Hasil Evaluasi metode SVM")
                # Hitung metrics
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(plt.show())

                st.write("Accuracy :")
                st.info(accuracy)

                st.write("Precision :")
                st.info(precision)

                st.write("Recall :")
                st.info(recall)

                st.write("F1-Score :")
                st.info(f1) 
            else:
                st.warning("Masukkan nilai dan data pada form")


    elif selected == "Implementation":
        with st.form("Implementation"):
            df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/Data_stunting_kalianget_sistem.csv')
            df = df.drop(columns=['Nama Lengkap'])
            x = df.drop(columns=['Status Stunting'])
            y = df['Status Stunting']

            #Normalisasi Dataset
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(x)
            features_names = x.columns.copy()
            #features_names.remove('label')
            scaled_features = pd.DataFrame(scaled, columns=features_names)

            # Fungsi untuk evaluasi kromosom
            def evaluate(individual, X_train, X_test, y_train, y_test):
                selected_features = [index for index, value in enumerate(individual) if value]

                if not selected_features:
                    return 0.0, 0, 0, 0, 0

                X_train_selected = X_train.iloc[:, selected_features].copy().to_numpy()
                X_test_selected = X_test.iloc[:, selected_features].copy().to_numpy()

                clf = SVC(C=3.0,gamma=0.1,kernel='rbf',max_iter=30)
                clf.fit(X_train_selected, y_train)

                y_pred = clf.predict(X_test_selected)

                accuracy = accuracy_score(y_test, y_pred)*100
                precision = precision_score(y_test, y_pred, average='weighted')*100
                recall = recall_score(y_test, y_pred, average='weighted')*100
                f1 = f1_score(y_test, y_pred, average='weighted')*100

                return accuracy, precision, recall, f1, len(selected_features),

            # Fungsi untuk menghasilkan populasi awal
            def generate_individual():
                return np.random.randint(0, 2, size=num_features).tolist()

            # Pisahkan data menjadi data latih dan data uji
            X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.5, random_state=42)

            # Jumlah fitur dalam dataset
            num_features = scaled_features.shape[1]

            # Input parameter GA
            num_generations = 20
            population_size = 10
            crossover_rate = 0.7
            mutation_rate = 0.2

            # Konfigurasi Algoritma Genetika
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMulti)
            toolbox = base.Toolbox()
            toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_rate)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Generate populasi awal
            population = toolbox.population(n=population_size)

            # Evaluasi populasi awal
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Evolusi populasi
            for generation in range(num_generations):
                # Pilih elit
                offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_rate, mutpb=mutation_rate)

                # Evaluasi offspring
                fitnesses = list(map(toolbox.evaluate, offspring))
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit

                # Ganti populasi dengan offspring
                population[:] = offspring

                # Pilih individu terbaik setelah evolusi
                best_individual = tools.selBest(population, k=1)[0]

                # Menggunakan fitur terpilih untuk pelatihan dan pengujian
                X_train_selected = X_train.iloc[:, [index for index, value in enumerate(best_individual) if value]].copy().to_numpy()
                X_test_selected = X_test.iloc[:, [index for index, value in enumerate(best_individual) if value]].copy().to_numpy()

                # Melatih model SVM dengan fitur terpilih
                clf = SVC(C=3.0,gamma=0.1,kernel='rbf',max_iter=30)
                clf.fit(X_train_selected, y_train)

                # Membuat prediksi
                y_pred = clf.predict(X_test_selected)


            st.write('### Prediksi Stunting Pada Balita')
            Nama_Lengkap = st.text_input('Masukkan Nama Lengkap Balita')  
            jenis_kelamin = st.selectbox('Masukkan Jenis Kelamin Balita',('Laki-laki','Perempuan'))
            umur_options = range(1, 61)
            umur = st.selectbox('Masukkan Umur Balita dalam (Bulan)', umur_options)
            BB_lahir = st.number_input('Masukkan Berat Badan Lahir Balita')
            PB_lahir = st.number_input('Masukkan Tinggi Badan Lahir Balita')
            BB_Saatini = st.number_input('Masukkan Berat Badan Balita Saat Ini')
            TB_Saatini = st.number_input('Masukkan Tinggi Badan Balita Saat Ini')
            

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                # Fungsi untuk menghitung Z_score dan menentukan Status
                def hitung_status_umur(option, umur, TB_Saatini):
                    if option == 'Laki-laki':
                        if umur == 0:
                            return None
                        elif umur == 1:
                            Z_score = (TB_Saatini - 54.7) / (54.7 - 52.8)
                        elif umur == 2:
                            Z_score = (TB_Saatini - 58.4) / (58.4 - 56.4)
                        elif umur == 3:
                            Z_score = (TB_Saatini - 61.4) / (61.4 - 59.4)
                        elif umur == 4:
                            Z_score = (TB_Saatini - 63.9) / (63.9 - 61.8)
                        elif umur == 5:
                            Z_score = (TB_Saatini - 65.9) / (65.9 - 63.8)
                        elif umur == 6:
                            Z_score = (TB_Saatini - 67.6) / (67.6 - 65.5)
                        elif umur == 7:
                            Z_score = (TB_Saatini - 69.2) / (69.2 - 67.0)
                        elif umur == 8:
                            Z_score = (TB_Saatini -70.6) / (70.6 - 68.4)
                        elif umur == 9:
                            Z_score = (TB_Saatini - 72.0) / (72.0 - 69.7)
                        elif umur == 10:
                            Z_score = (TB_Saatini - 73.3) / (73.3 - 71.0)
                        elif umur == 11:
                            Z_score = (TB_Saatini - 74.5) / (74.5 - 72.2)
                        elif umur == 12:
                            Z_score = (TB_Saatini - 75.7) / (75.7 - 73.4)
                        elif umur == 13:
                            Z_score = (TB_Saatini - 76.9) / (76.9 - 74.5)
                        elif umur == 14:
                            Z_score = (TB_Saatini - 78.0) / (78.0 - 75.6)
                        elif umur == 15:
                            Z_score = (TB_Saatini - 79.1) / (79.1 - 76.6)
                        elif umur == 16:
                            Z_score = (TB_Saatini - 80.2) / (80.2 - 77.6)
                        elif umur == 17:
                            Z_score = (TB_Saatini - 81.2) / (81.2 - 78.6)
                        elif umur == 18:
                            Z_score = (TB_Saatini - 82.3) / (82.3 - 79.6)
                        elif umur == 19:
                            Z_score = (TB_Saatini - 83.2) / (83.2 - 80.5)
                        elif umur == 20:
                            Z_score = (TB_Saatini - 84.2) / (84.2 - 81.4)
                        elif umur == 21:
                            Z_score = (TB_Saatini - 85.1) / (85.1 - 82.3)
                        elif umur == 22:
                            Z_score = (TB_Saatini - 86.0) / (86.0 - 83.1)
                        elif umur == 23:
                            Z_score = (TB_Saatini - 86.9) / (86.9 - 83.9)
                        elif umur == 24:
                            Z_score = (TB_Saatini - 87.8) / (87.8 - 84.8)
                        elif umur == 25:
                            Z_score = (TB_Saatini - 88.0) / (88.0 - 84.9)
                        elif umur == 26:
                            Z_score = (TB_Saatini - 88.8) / (88.8 - 85.6)
                        elif umur == 27:
                            Z_score = (TB_Saatini - 89.6) / (89.6 - 86.4)
                        elif umur == 28:
                            Z_score = (TB_Saatini - 90.4) / (90.4 - 87.1)
                        elif umur == 29:
                            Z_score = (TB_Saatini - 91.2) / (91.2 - 87.8)
                        elif umur == 30:
                            Z_score = (TB_Saatini - 91.9) / (91.9 - 88.5)
                        elif umur == 31:
                            Z_score = (TB_Saatini - 92.7) / (92.7 - 89.2)
                        elif umur == 32:
                            Z_score = (TB_Saatini - 93.4) / (93.4 - 89.9)
                        elif umur == 33:
                            Z_score = (TB_Saatini - 94.1) / (94.1 - 90.5)
                        elif umur == 34:
                            Z_score = (TB_Saatini - 94.8) / (94.8 - 91.1)
                        elif umur == 35:
                            Z_score = (TB_Saatini - 95.4) / (95.4 - 91.8)
                        elif umur == 36:
                            Z_score = (TB_Saatini - 96.1) / (96.1 - 92.4)
                        elif umur == 37:
                            Z_score = (TB_Saatini - 96.7) / (96.7 - 93.0)
                        elif umur == 38:
                            Z_score = (TB_Saatini - 97.4) / (97.4 - 93.6)
                        elif umur == 39:
                            Z_score = (TB_Saatini - 98.0) / (98.0 - 94.2)
                        elif umur == 40:
                            Z_score = (TB_Saatini - 98.6) / (98.6 - 94.7)
                        elif umur == 41:
                            Z_score = (TB_Saatini - 99.2) / (99.2 - 95.3)
                        elif umur == 42:
                            Z_score = (TB_Saatini - 99.9) / (99.9 - 95.9)
                        elif umur == 43:
                            Z_score = (TB_Saatini - 100.4) / (100.4 - 96.4)
                        elif umur == 44:
                            Z_score = (TB_Saatini - 101.0) / (101.0 - 97.0)
                        elif umur == 45:
                            Z_score = (TB_Saatini - 101.6) / (101.6 - 97.5)
                        elif umur == 46:
                            Z_score = (TB_Saatini - 102.2) / (102.2 - 98.1)
                        elif umur == 47:
                            Z_score = (TB_Saatini - 102.8) / (102.8 - 98.6)
                        elif umur == 48:
                            Z_score = (TB_Saatini - 103.3) / (103.3 - 99.1)
                        elif umur == 59:
                            Z_score = (TB_Saatini - 103.9) / (103.9 - 99.7)
                        elif umur == 50:
                            Z_score = (TB_Saatini - 104.4) / (104.4 - 100.2)
                        elif umur == 51:
                            Z_score = (TB_Saatini - 105.0) / (105.0 - 100.7)
                        elif umur == 52:
                            Z_score = (TB_Saatini - 105.6) / (105.6 - 101.2)
                        elif umur == 53:
                            Z_score = (TB_Saatini - 106.1) / (106.1  - 101.7)
                        elif umur == 54:
                            Z_score = (TB_Saatini - 106.7) / (106.7 - 102.3)
                        elif umur == 55:
                            Z_score = (TB_Saatini - 107.2) / (107.2 - 102.8)
                        elif umur == 56:
                            Z_score = (TB_Saatini - 107.8) / (107.8 - 103.3)
                        elif umur == 57:
                            Z_score = (TB_Saatini - 108.3) / (108.3 - 103.8)
                        elif umur == 58:
                            Z_score = (TB_Saatini - 108.9) / (108.9 - 104.3)
                        elif umur == 59:
                            Z_score = (TB_Saatini - 109.4) / (109.4 - 104.8)
                        elif umur == 60:
                            Z_score = (TB_Saatini - 110.0) / (110.0 -105.3)
                        else:
                            Z_score = (TB_Saatini - 110.0) / (110.0 -105.3)

                        # Tentukan Status berdasarkan Z_score
                        if Z_score < -3:
                            return "Sangat Pendek"
                        elif -3 < Z_score < -2:
                            return "Pendek"
                        elif -2 < Z_score < 3:
                            return "Normal"
                        else:
                            return "Tinggi"
                            
                    #Perempuan
                    elif option == 'Perempuan':
                        if umur == 0:
                            return None
                        elif umur == 1:
                            Z_score = (TB_Saatini - 53.7) / (53.7 - 51.7)
                        elif umur == 2:
                            Z_score = (TB_Saatini - 57.1) / (57.1 - 55.0)
                        elif umur == 3:
                            Z_score = (TB_Saatini - 59.8) / (59.8 - 57.0)
                        elif umur == 4:
                            Z_score = (TB_Saatini - 62.1) / (62.1 - 59.9)
                        elif umur == 5:
                            Z_score = (TB_Saatini - 64.0) / (64.0 - 61.8)
                        elif umur == 6:
                            Z_score = (TB_Saatini - 65.7) / (65.7 - 63.5)
                        elif umur == 7:
                            Z_score = (TB_Saatini - 67.3) / (67.3 - 65.0)
                        elif umur == 8:
                            Z_score = (TB_Saatini - 68.7) / (68.7 - 66.4)
                        elif umur == 9:
                            Z_score = (TB_Saatini - 70.1) / (70.1 - 67.7)
                        elif umur == 10:
                            Z_score = (TB_Saatini - 71.5) / (71.5 - 69.0)
                        elif umur == 11:
                            Z_score = (TB_Saatini - 72.8) / (72.8 - 70.3)
                        elif umur == 12:
                            Z_score = (TB_Saatini - 74.0) / (74.0 - 71.4)
                        elif umur == 13:
                            Z_score = (TB_Saatini - 75.2) / (75.2 - 72.6)
                        elif umur == 14:
                            Z_score = (TB_Saatini - 76.4) / (76.4 - 73.7)
                        elif umur == 15:
                            Z_score = (TB_Saatini - 77.5) / (77.5 - 74.8)
                        elif umur == 16:
                            Z_score = (TB_Saatini - 78.6) / (78.6 - 75.8)
                        elif umur == 17:
                            Z_score = (TB_Saatini - 79.7) / (79.7 - 76.8)
                        elif umur == 18:
                            Z_score = (TB_Saatini - 80.7) / (80.7 - 77.8)
                        elif umur == 19:
                            Z_score = (TB_Saatini - 81.7) / (81.7 - 78.8)
                        elif umur == 20:
                            Z_score = (TB_Saatini - 82.7) / (82.7 - 79.7)
                        elif umur == 21:
                            Z_score = (TB_Saatini - 83.7)  / (83.7 - 80.6)
                        elif umur == 22:
                            Z_score = (TB_Saatini - 84.6) / (84.6 - 81.5)
                        elif umur == 23:
                            Z_score = (TB_Saatini - 53.7) / (53.7 - 82.3)
                        #elif umur == 24:
                            Z_score = (TB_Saatini - 86.4) / (86.4 - 83.2)
                        elif umur == 25:
                            Z_score = (TB_Saatini - 86.6) / (86.6 - 83.3)
                        elif umur == 26:
                            Z_score = (TB_Saatini - 87.4) / (87.4 - 84.1)
                        elif umur == 27:
                            Z_score = (TB_Saatini - 84.9) / (88.3 - 84.9)
                        elif umur == 28:
                            Z_score = (TB_Saatini - 89.1) / (89.1 - 85.7)
                        elif umur == 29:
                            Z_score = (TB_Saatini - 89.9) / (89.9 - 86.4)
                        elif umur == 30:
                            Z_score = (TB_Saatini - 90.7) / (90.7 - 87.1)
                        elif umur == 31:
                            Z_score = (TB_Saatini - 91.4) / (91.4 - 87.9)
                        elif umur == 32:
                            Z_score = (TB_Saatini - 92.2) / (92.2 - 88.6)
                        elif umur == 33:
                            Z_score = (TB_Saatini - 92.9) / (92.9 - 89.3)
                        elif umur == 34:
                            Z_score = (TB_Saatini - 93.6) / (93.6 - 89.9)
                        elif umur == 35:
                            Z_score = (TB_Saatini - 94.4) / (94.4 - 90.6)
                        elif umur == 36:
                            Z_score = (TB_Saatini - 95.1) / (95.1 - 91.2)
                        elif umur == 37:
                            Z_score = (TB_Saatini - 95.7) / (95.7 - 91.9)
                        elif umur == 38:
                            Z_score = (TB_Saatini - 96.4) / (96.4 - 92.5)
                        elif umur == 39:
                            Z_score = (TB_Saatini - 97.1) / (97.1 - 93.1)
                        elif umur == 40:
                            Z_score = (TB_Saatini - 97.7) / (97.7 - 93.8)
                        elif umur == 41:
                            Z_score = (TB_Saatini - 98.4) / (98.4 - 94.4)
                        elif umur == 42:
                            Z_score = (TB_Saatini - 99.0) / (99.0 - 95.0)
                        elif umur == 43:
                            Z_score = (TB_Saatini - 99.7) / (99.7 - 95.6)
                        elif umur == 44:
                            Z_score = (TB_Saatini - 100.3) / (100.3 - 96.2)
                        elif umur == 45:
                            Z_score = (TB_Saatini - 100.9) / (100.9 - 96.7)
                        elif umur == 46:
                            Z_score = (TB_Saatini - 101.5) / (101.5 - 97.3)
                        elif umur == 47:
                            Z_score = (TB_Saatini - 102.1) / (102.1 - 97.9)
                        elif umur == 48:
                            Z_score = (TB_Saatini - 102.7) / (102.7 - 98.4)
                        elif umur == 49:
                            Z_score = (TB_Saatini - 103.3) / (103.3 - 99.0)
                        elif umur == 50:
                            Z_score = (TB_Saatini - 103.9) / (103.9 - 99.5)
                        elif umur == 51:
                            Z_score = (TB_Saatini - 104.5 ) / (104.5 - 100.1)
                        elif umur == 52:
                            Z_score = (TB_Saatini - 105.0) / (105.0 - 100.6)
                        elif umur == 53:
                            Z_score = (TB_Saatini - 105.6) / (105.6 - 101.1)
                        elif umur == 54:
                            Z_score = (TB_Saatini - 106.2) / (106.2 - 101.6)
                        elif umur == 55:
                            Z_score = (TB_Saatini - 106.7) / (106.7 - 102.2)
                        elif umur == 56:
                            Z_score = (TB_Saatini - 107.3) / (107.3 - 102.7)
                        elif umur == 57:
                            Z_score = (TB_Saatini - 107.8) / (107.8 - 103.2)
                        elif umur == 58:
                            Z_score = (TB_Saatini - 108.4) / (108.4 - 103.7)
                        elif umur == 59:
                            Z_score = (TB_Saatini - 108.9) / (108.9 - 104.2)
                        elif umur == 60:
                            Z_score = (TB_Saatini - 109.4) / (109.4 - 104.7)
                        else:
                            return None  # Umur di luar rentang yang ditangani

                        # Tentukan Status berdasarkan Z_score
                        if Z_score < -3:
                            return "Sangat Pendek"
                        elif -3 < Z_score < -2:
                            return "Pendek"
                        elif -2 < Z_score < 3:
                            return "Normal"
                        else:
                            return "Tinggi"
                    else:
                        return None  # Kondisi untuk jenis kelamin selain 'Laki-laki' tidak ditangani
                    
                status = hitung_status_umur(jenis_kelamin, umur, TB_Saatini)

                # Menampilkan Status jika hasil tidak None
                if status is not None:
                    if status == 'Normal':
                        status_gizi = 1
                    elif status == 'Pendek':
                        status_gizi = 2
                    elif status == 'Sangat Pendek':
                        status_gizi = 3
                    elif status == 'Tinggi':
                        status_gizi = 4  

                    if jenis_kelamin == 'Laki-laki':
                        jk = 1
                    else:
                        jk = 0
                    
                    inputs = np.array([
                        jk,
                        umur,
                        BB_lahir ,
                        PB_lahir,
                        BB_Saatini,
                        TB_Saatini,
                        status_gizi
                    ])

                    st.write("#### Status Gizi TB/U")
                    st.write(status)

                    #Data Input
                    st.write("#### Data Input")
                    st.dataframe(inputs)

                    #Normalisasi data input
                    df_min = x.iloc[:,0:7].min()
                    df_max = x.iloc[:,0:7].max()
                    input_norm = ((inputs - df_min) / (df_max - df_min))
                    input_norm = pd.DataFrame(np.array(input_norm).reshape(1,-1))

                    st.write("#### Normalisasi data Input")
                    st.write(input_norm)

                    #Prediksi
                    selected_features_indices = [index for index, value in enumerate(best_individual) if value]
                    selected_feature_names = [features_names[i] for i in selected_features_indices]

                    st.write("#### Fitur terseleksi")
                    st.info(selected_feature_names)
                    new_data_selected = input_norm.iloc[:, selected_features_indices].copy().to_numpy()
                    new_predictions = clf.predict(new_data_selected)
                    
                    #Hasil Prediksi
                    st.write('#### Hasil Prediksi Stunting')
                    if new_predictions == 1:
                        st.success("Normal")
                    if new_predictions == -1:
                        st.error("Stunting")

