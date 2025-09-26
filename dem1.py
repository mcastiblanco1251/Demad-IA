import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import datetime as dt
from dateutil.relativedelta import relativedelta
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Pronóstico y Gestión de Inventarios",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
image1 = Image.open('logo.png')
row1_1, row1_2= st.columns((1, 3))
# with row1_1:
#         imagelp2 = Image.open('predictor2.png')
#         new_image5=imagelp2.resize((120, 40))
#         st.image(new_image5)
#         st.markdown('Dev App by [Smart](https://ia.saineco.com//)')
#         #st.image(new_image2)

with row1_2:
        imagelp = Image.open('predictor.png')
        imagelp2 = Image.open('predictor2.png')
        new_image3=imagelp.resize((240, 80))
        new_image4=imagelp2.resize((240, 80))
        # Título principal
        st.title("🏭 Sistema de Pronóstico de Demanda y Gestión de Inventarios")
        st.markdown("### Empresa de Alimentos - Gestión Inteligente de Compras")

with row1_1:
        st.image(image1)
        st.markdown('Dev App by [SAINECO](https://ia.saineco.com//)')

# Inicializar datos de ejemplo si no existen en session_state
if 'initialized' not in st.session_state:
    
    # Productos (10 productos)
    productos = [f"Producto_{i+1}" for i in range(10)]
    
    # Insumos (50 insumos)
    insumos = [f"Insumo_{i+1}" for i in range(50)]
    
    # Generar datos históricos de demanda (últimos 24 meses)
    fechas = pd.date_range(end=dt.date.today(), periods=24, freq='M')
    np.random.seed(42)
    
    demanda_historica = []
    for fecha in fechas:
        for producto in productos:
            demanda = np.random.normal(1000, 200) + np.sin(fecha.month * 2 * np.pi / 12) * 100
            demanda = max(0, demanda)
            demanda_historica.append({
                'fecha': fecha,
                'producto': producto,
                'demanda': round(demanda, 0)
            })
    
    st.session_state.demanda_df = pd.DataFrame(demanda_historica)
    
    # BOM (Bill of Materials) - Relación producto-insumo
    bom_data = []
    np.random.seed(42)
    for producto in productos:
        # Cada producto usa entre 3-8 insumos
        num_insumos = np.random.randint(3, 9)
        insumos_producto = np.random.choice(insumos, num_insumos, replace=False)
        
        for insumo in insumos_producto:
            cantidad = np.random.uniform(0.1, 5.0)
            bom_data.append({
                'producto': producto,
                'insumo': insumo,
                'cantidad_por_unidad': round(cantidad, 3)
            })
    
    st.session_state.bom_df = pd.DataFrame(bom_data)
    
    # Inventario actual
    inventario_data = []
    for insumo in insumos:
        stock_planta = np.random.randint(100, 2000)
        stock_transito = np.random.randint(0, 500)
        stock_minimo = np.random.randint(50, 200)
        inventario_data.append({
            'insumo': insumo,
            'stock_planta': stock_planta,
            'stock_transito': stock_transito,
            'stock_minimo': stock_minimo,
            'costo_unitario': round(np.random.uniform(1.0, 50.0), 2),
            'tiempo_entrega_dias': np.random.randint(5, 30)
        })
    
    st.session_state.inventario_df = pd.DataFrame(inventario_data)
    st.session_state.initialized = True

# Sidebar para navegación
st.sidebar.title("📋 Menú Principal")
menu = st.sidebar.selectbox(
    "Selecciona una opción:",
    ["🏠 Dashboard Principal", "📈 Pronóstico de Demanda", "🔧 Configuración BOM", 
     "📦 Gestión de Inventario", "🛒 Planificación de Compras", "⚙️ Optimización"]
)

def entrenar_modelo_pronostico(df_demanda, producto):
    """Entrenar modelo de pronóstico para un producto específico"""
    data = df_demanda[df_demanda['producto'] == producto].copy()
    data = data.sort_values('fecha')
    
    # Crear características temporales
    data['mes'] = data['fecha'].dt.month
    data['trimestre'] = data['fecha'].dt.quarter
    data['año'] = data['fecha'].dt.year
    data['tendencia'] = range(len(data))
    
    # Lag features
    data['demanda_lag1'] = data['demanda'].shift(1)
    data['demanda_lag2'] = data['demanda'].shift(2)
    data['media_movil_3'] = data['demanda'].rolling(window=3).mean()
    
    # Eliminar filas con NaN
    data = data.dropna()
    
    if len(data) < 10:
        return None, None
    
    # Preparar datos para entrenamiento
    features = ['mes', 'trimestre', 'tendencia', 'demanda_lag1', 'demanda_lag2', 'media_movil_3']
    X = data[features]
    y = data['demanda']
    
    # Entrenar modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    return modelo, features

def generar_pronostico(modelo, features, ultimo_registro, meses_adelante=3):
    """Generar pronóstico para los próximos meses"""
    if modelo is None:
        return []
    
    pronosticos = []
    fecha_actual = ultimo_registro['fecha']
    
    for i in range(meses_adelante):
        fecha_nueva = fecha_actual + relativedelta(months=i+1)
        
        # Crear características para el pronóstico
        nueva_fila = {
            'mes': fecha_nueva.month,
            'trimestre': fecha_nueva.quarter,
            'tendencia': ultimo_registro['tendencia'] + i + 1,
            'demanda_lag1': ultimo_registro['demanda'] if i == 0 else pronosticos[i-1],
            'demanda_lag2': ultimo_registro['demanda_lag1'] if i == 0 else (ultimo_registro['demanda'] if i == 1 else pronosticos[i-2]),
            'media_movil_3': ultimo_registro['media_movil_3'] if i == 0 else np.mean(pronosticos[max(0, i-2):i+1]) if i > 0 else ultimo_registro['media_movil_3']
        }
        
        X_nuevo = pd.DataFrame([nueva_fila])[features]
        prediccion = modelo.predict(X_nuevo)[0]
        prediccion = max(0, prediccion)  # No permitir demanda negativa
        
        pronosticos.append(round(prediccion, 0))
        
        # Actualizar último registro para la siguiente iteración
        ultimo_registro = {
            'fecha': fecha_nueva,
            'demanda': prediccion,
            'demanda_lag1': ultimo_registro['demanda'] if i == 0 else pronosticos[i-1],
            'tendencia': ultimo_registro['tendencia'] + i + 1,
            'media_movil_3': nueva_fila['media_movil_3']
        }
    
    return pronosticos

if menu == "🏠 Dashboard Principal":
    st.header("Dashboard Ejecutivo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_productos = len(st.session_state.demanda_df['producto'].unique())
        st.metric("Productos", total_productos)
    
    with col2:
        total_insumos = len(st.session_state.inventario_df)
        st.metric("Insumos", total_insumos)
    
    with col3:
        valor_inventario = (st.session_state.inventario_df['stock_planta'] * 
                           st.session_state.inventario_df['costo_unitario']).sum()
        st.metric("Valor Inventario", f"${valor_inventario:,.0f}")
    
    with col4:
        insumos_criticos = len(st.session_state.inventario_df[
            st.session_state.inventario_df['stock_planta'] <= st.session_state.inventario_df['stock_minimo']
        ])
        st.metric("Insumos Críticos", insumos_criticos, delta=-insumos_criticos if insumos_criticos > 0 else 0)
    
    # Gráfico de demanda histórica
    st.subheader("📊 Tendencia de Demanda por Producto")
    demanda_mensual = st.session_state.demanda_df.groupby(['fecha', 'producto'])['demanda'].sum().reset_index()
    
    fig = px.line(demanda_mensual, x='fecha', y='demanda', color='producto',
                  title="Demanda Histórica por Producto")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Estado del inventario
    st.subheader("📦 Estado Actual del Inventario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Insumos con stock crítico
        stock_critico = st.session_state.inventario_df[
            st.session_state.inventario_df['stock_planta'] <= st.session_state.inventario_df['stock_minimo']
        ]
        
        if not stock_critico.empty:
            st.warning(f"⚠️ {len(stock_critico)} insumos en stock crítico")
            st.dataframe(stock_critico[['insumo', 'stock_planta', 'stock_minimo']], hide_index=True)
        else:
            st.success("✅ Todos los insumos tienen stock adecuado")
    
    with col2:
        # Top 10 insumos por valor
        top_insumos = st.session_state.inventario_df.copy()
        top_insumos['valor_total'] = top_insumos['stock_planta'] * top_insumos['costo_unitario']
        top_insumos = top_insumos.nlargest(10, 'valor_total')
        
        fig = px.bar(top_insumos, x='insumo', y='valor_total',
                     title="Top 10 Insumos por Valor en Inventario")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

elif menu == "📈 Pronóstico de Demanda":
    st.header("Pronóstico de Demanda con IA")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuración")
        producto_seleccionado = st.selectbox(
            "Selecciona un producto:",
            st.session_state.demanda_df['producto'].unique()
        )
        
        meses_pronostico = st.slider("Meses a pronosticar:", 1, 12, 3)
        
        if st.button("🚀 Generar Pronóstico", type="primary"):
            with st.spinner("Entrenando modelo de IA..."):
                # Preparar datos para el producto seleccionado
                df_producto = st.session_state.demanda_df[
                    st.session_state.demanda_df['producto'] == producto_seleccionado
                ].copy()
                df_producto = df_producto.sort_values('fecha')
                
                # Crear características adicionales
                df_producto['mes'] = df_producto['fecha'].dt.month
                df_producto['trimestre'] = df_producto['fecha'].dt.quarter
                df_producto['año'] = df_producto['fecha'].dt.year
                df_producto['tendencia'] = range(len(df_producto))
                df_producto['demanda_lag1'] = df_producto['demanda'].shift(1)
                df_producto['demanda_lag2'] = df_producto['demanda'].shift(2)
                df_producto['media_movil_3'] = df_producto['demanda'].rolling(window=3).mean()
                df_producto = df_producto.dropna()
                
                # Entrenar modelo
                modelo, features = entrenar_modelo_pronostico(st.session_state.demanda_df, producto_seleccionado)
                
                if modelo is not None:
                    ultimo_registro = df_producto.iloc[-1].to_dict()
                    pronosticos = generar_pronostico(modelo, features, ultimo_registro, meses_pronostico)
                    
                    # Guardar en session_state
                    st.session_state.ultimo_pronostico = {
                        'producto': producto_seleccionado,
                        'pronosticos': pronosticos,
                        'meses': meses_pronostico,
                        'modelo': modelo,
                        'features': features
                    }
                    
                    st.success("✅ Pronóstico generado exitosamente")
                else:
                    st.error("❌ No hay suficientes datos para entrenar el modelo")
    
    with col2:
        st.subheader("📊 Resultados del Pronóstico")
        
        if 'ultimo_pronostico' in st.session_state and st.session_state.ultimo_pronostico['producto'] == producto_seleccionado:
            # Mostrar pronósticos
            pronostico_data = st.session_state.ultimo_pronostico
            
            # Crear DataFrame para visualización
            fechas_futuras = pd.date_range(
                start=st.session_state.demanda_df['fecha'].max() + relativedelta(months=1),
                periods=pronostico_data['meses'],
                freq='M'
            )
            
            df_pronostico = pd.DataFrame({
                'fecha': fechas_futuras,
                'demanda_pronosticada': pronostico_data['pronosticos'],
                'tipo': 'Pronóstico'
            })
            
            # Datos históricos
            df_historico = st.session_state.demanda_df[
                st.session_state.demanda_df['producto'] == producto_seleccionado
            ].copy()
            df_historico['tipo'] = 'Histórico'
            df_historico = df_historico.rename(columns={'demanda': 'demanda_pronosticada'})
            
            # Combinar datos
            df_completo = pd.concat([
                df_historico[['fecha', 'demanda_pronosticada', 'tipo']],
                df_pronostico
            ])
            
            # Gráfico
            fig = px.line(df_completo, x='fecha', y='demanda_pronosticada', color='tipo',
                         title=f"Pronóstico de Demanda - {producto_seleccionado}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de pronósticos
            st.subheader("📋 Pronósticos Detallados")
            df_tabla = pd.DataFrame({
                'Mes': fechas_futuras.strftime('%B %Y'),
                'Demanda Pronosticada': [f"{x:,.0f} unidades" for x in pronostico_data['pronosticos']]
            })
            st.dataframe(df_tabla, hide_index=True)
            #st.dataframe(df_historico, hide_index=True)
            #st.dataframe(pronostico_data, hide_index=True)
            # Métricas del modelo
            if len(df_historico) > 10:
                # Calcular métricas en los últimos datos
                X_test = df_historico.iloc[-5:]#[pronostico_data['features']]
                X_test['mes'] = X_test['fecha'].dt.month
                X_test['trimestre'] = X_test['fecha'].dt.quarter
                X_test['año'] = X_test['fecha'].dt.year
                X_test['tendencia'] = range(len(X_test))

                # Lag features
                X_test['demanda_lag1'] = X_test['demanda_pronosticada'].shift(1)
                X_test['demanda_lag2'] = X_test['demanda_pronosticada'].shift(2)
                X_test['media_movil_3'] = X_test['demanda_pronosticada'].rolling(window=3).mean()
                
                y_test = df_historico.iloc[-5:]['demanda_pronosticada']
                
                model1, features= entrenar_modelo_pronostico(st.session_state.demanda_df, producto_seleccionado)
                y_pred = model1.predict(X_test[features])
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("MAE (Error Absoluto Medio)", f"{mae:.0f}")
                with col_m2:
                    st.metric("RMSE (Error Cuadrático Medio)", f"{rmse:.0f}")

elif menu == "🔧 Configuración BOM":
    st.header("Bill of Materials (BOM)")
    st.markdown("Configuración de insumos necesarios por producto")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("➕ Agregar/Modificar BOM")
        
        with st.form("bom_form"):
            producto = st.selectbox("Producto:", st.session_state.demanda_df['producto'].unique())
            insumo = st.selectbox("Insumo:", st.session_state.inventario_df['insumo'].unique())
            cantidad = st.number_input("Cantidad por unidad:", min_value=0.001, value=1.0, step=0.001)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                agregar = st.form_submit_button("➕ Agregar", type="primary")
            with col_btn2:
                eliminar = st.form_submit_button("🗑️ Eliminar")
        
        if agregar:
            # Verificar si ya existe la combinación
            exists = ((st.session_state.bom_df['producto'] == producto) & 
                     (st.session_state.bom_df['insumo'] == insumo)).any()
            
            if exists:
                # Actualizar cantidad
                st.session_state.bom_df.loc[
                    (st.session_state.bom_df['producto'] == producto) & 
                    (st.session_state.bom_df['insumo'] == insumo),
                    'cantidad_por_unidad'
                ] = cantidad
                st.success(f"✅ Actualizado: {producto} - {insumo}")
            else:
                # Agregar nueva entrada
                nueva_fila = pd.DataFrame({
                    'producto': [producto],
                    'insumo': [insumo],
                    'cantidad_por_unidad': [cantidad]
                })
                st.session_state.bom_df = pd.concat([st.session_state.bom_df, nueva_fila], ignore_index=True)
                st.success(f"✅ Agregado: {producto} - {insumo}")
        
        if eliminar:
            # Eliminar entrada
            st.session_state.bom_df = st.session_state.bom_df[
                ~((st.session_state.bom_df['producto'] == producto) & 
                  (st.session_state.bom_df['insumo'] == insumo))
            ]
            st.success(f"🗑️ Eliminado: {producto} - {insumo}")
    
    with col2:
        st.subheader("📋 BOM Actual")
        
        # Filtro por producto
        producto_filtro = st.selectbox("Filtrar por producto:", 
                                      ['Todos'] + list(st.session_state.bom_df['producto'].unique()),
                                      key="bom_filter")
        
        if producto_filtro == 'Todos':
            bom_display = st.session_state.bom_df
        else:
            bom_display = st.session_state.bom_df[st.session_state.bom_df['producto'] == producto_filtro]
        
        st.dataframe(bom_display, hide_index=True, height=400)
        
        # Estadísticas
        st.subheader("📊 Estadísticas BOM")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            productos_con_bom = len(st.session_state.bom_df['producto'].unique())
            st.metric("Productos con BOM", productos_con_bom)
        
        with col_s2:
            insumos_utilizados = len(st.session_state.bom_df['insumo'].unique())
            st.metric("Insumos Utilizados", insumos_utilizados)

elif menu == "📦 Gestión de Inventario":
    st.header("Gestión de Inventario")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Actualizar Inventario")
        
        with st.form("inventario_form"):
            insumo_sel = st.selectbox("Insumo:", st.session_state.inventario_df['insumo'].unique())
            
            # Obtener valores actuales
            valores_actuales = st.session_state.inventario_df[
                st.session_state.inventario_df['insumo'] == insumo_sel
            ].iloc[0]
            
            stock_planta = st.number_input("Stock en Planta:", 
                                         value=int(valores_actuales['stock_planta']),
                                         min_value=0)
            stock_transito = st.number_input("Stock en Tránsito:", 
                                           value=int(valores_actuales['stock_transito']),
                                           min_value=0)
            stock_minimo = st.number_input("Stock Mínimo:", 
                                         value=int(valores_actuales['stock_minimo']),
                                         min_value=0)
            costo_unitario = st.number_input("Costo Unitario:", 
                                           value=float(valores_actuales['costo_unitario']),
                                           min_value=0.01)
            tiempo_entrega = st.number_input("Tiempo de Entrega (días):", 
                                           value=int(valores_actuales['tiempo_entrega_dias']),
                                           min_value=1)
            
            actualizar = st.form_submit_button("🔄 Actualizar", type="primary")
        
        if actualizar:
            # Actualizar valores
            idx = st.session_state.inventario_df[
                st.session_state.inventario_df['insumo'] == insumo_sel
            ].index[0]
            
            st.session_state.inventario_df.loc[idx, 'stock_planta'] = stock_planta
            st.session_state.inventario_df.loc[idx, 'stock_transito'] = stock_transito
            st.session_state.inventario_df.loc[idx, 'stock_minimo'] = stock_minimo
            st.session_state.inventario_df.loc[idx, 'costo_unitario'] = costo_unitario
            st.session_state.inventario_df.loc[idx, 'tiempo_entrega_dias'] = tiempo_entrega
            
            st.success(f"✅ Inventario actualizado para {insumo_sel}")
    
    with col2:
        st.subheader("📊 Estado del Inventario")
        
        # Calcular métricas
        inventario_display = st.session_state.inventario_df.copy()
        inventario_display['stock_total'] = inventario_display['stock_planta'] + inventario_display['stock_transito']
        inventario_display['valor_total'] = inventario_display['stock_total'] * inventario_display['costo_unitario']
        inventario_display['estado'] = inventario_display.apply(
            lambda row: 'Crítico' if row['stock_planta'] <= row['stock_minimo'] else 'Normal',
            axis=1
        )
        
        # Filtros
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filtro_estado = st.selectbox("Filtrar por estado:", ['Todos', 'Crítico', 'Normal'])
        with col_f2:
            ordenar_por = st.selectbox("Ordenar por:", 
                                     ['insumo', 'stock_planta', 'valor_total', 'tiempo_entrega_dias'])
        
        # Aplicar filtros
        if filtro_estado != 'Todos':
            inventario_display = inventario_display[inventario_display['estado'] == filtro_estado]
        
        inventario_display = inventario_display.sort_values(ordenar_por, ascending=False)
        
        # Mostrar tabla
        columnas_mostrar = ['insumo', 'stock_planta', 'stock_transito', 'stock_total', 
                           'stock_minimo', 'valor_total', 'tiempo_entrega_dias', 'estado']
        st.dataframe(
            inventario_display[columnas_mostrar].round(2),
            hide_index=True,
            height=400
        )
        
        # Gráfico de valor por insumo
        fig = px.treemap(inventario_display.head(20), 
                        path=['estado', 'insumo'], 
                        values='valor_total',
                        title="Valor del Inventario por Insumo (Top 20)")
        st.plotly_chart(fig, use_container_width=True)

elif menu == "🛒 Planificación de Compras":
    st.header("Planificación Inteligente de Compras")
    
    if 'ultimo_pronostico' not in st.session_state:
        st.warning("⚠️ Primero debes generar un pronóstico de demanda en la sección correspondiente.")
        st.stop()
    
    st.subheader("📋 Plan de Compras Basado en Pronóstico")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuración")
        
        # Seleccionar productos para el plan
        productos_disponibles = list(st.session_state.demanda_df['producto'].unique())
        productos_seleccionados = st.multiselect(
            "Productos a incluir en el plan:",
            productos_disponibles,
            default=[st.session_state.ultimo_pronostico['producto']]
        )
        
        horizonte_meses = st.slider("Horizonte de planificación (meses):", 1, 6, 3)
        factor_seguridad = st.slider("Factor de seguridad:", 1.0, 2.0, 1.2, 0.1)
        
        if st.button("🔄 Calcular Plan de Compras", type="primary"):
            plan_compras = []
            
            for producto in productos_seleccionados:
                # Generar o usar pronóstico existente
                if producto == st.session_state.ultimo_pronostico['producto']:
                    pronosticos = st.session_state.ultimo_pronostico['pronosticos'][:horizonte_meses]
                else:
                    # Generar pronóstico para este producto
                    modelo, features = entrenar_modelo_pronostico(st.session_state.demanda_df, producto)
                    if modelo is not None:
                        df_producto = st.session_state.demanda_df[
                            st.session_state.demanda_df['producto'] == producto
                        ].sort_values('fecha')
                        
                        ultimo_registro = df_producto.iloc[-1]
                        ultimo_dict = {
                            'fecha': ultimo_registro['fecha'],
                            'demanda': ultimo_registro['demanda'],
                            'tendencia': len(df_producto),
                            'demanda_lag1': df_producto.iloc[-2]['demanda'] if len(df_producto) > 1 else ultimo_registro['demanda'],
                            'media_movil_3': df_producto['demanda'].tail(3).mean()
                        }
                        
                        pronosticos = generar_pronostico(modelo, features, ultimo_dict, horizonte_meses)
                    else:
                        continue
                
                # Calcular necesidades de insumos
                bom_producto = st.session_state.bom_df[st.session_state.bom_df['producto'] == producto]
                
                for _, bom_row in bom_producto.iterrows():
                    insumo = bom_row['insumo']
                    cantidad_por_unidad = bom_row['cantidad_por_unidad']
                    
                    # Calcular demanda total de insumo
                    demanda_total_insumo = sum(pronosticos) * cantidad_por_unidad * factor_seguridad
                    
                    # Obtener inventario actual
                    inventario_actual = st.session_state.inventario_df[
                        st.session_state.inventario_df['insumo'] == insumo
                    ].iloc[0]
                    
                    stock_disponible = inventario_actual['stock_planta'] + inventario_actual['stock_transito']
                    necesidad_compra = max(0, demanda_total_insumo - stock_disponible + inventario_actual['stock_minimo'])
                    
                    if necesidad_compra > 0:
                        plan_compras.append({
                            'producto': producto,
                            'insumo': insumo,
                            'demanda_pronosticada': sum(pronosticos),
                            'cantidad_por_unidad': cantidad_por_unidad,
                            'necesidad_insumo': demanda_total_insumo,
                            'stock_disponible': stock_disponible,
                            'stock_minimo': inventario_actual['stock_minimo'],
                            'cantidad_comprar': necesidad_compra,
                            'costo_unitario': inventario_actual['costo_unitario'],
                            'costo_total': necesidad_compra * inventario_actual['costo_unitario'],
                            'tiempo_entrega': inventario_actual['tiempo_entrega_dias']
                        })
            
            if plan_compras:
                st.session_state.plan_compras = pd.DataFrame(plan_compras)
                st.success("✅ Plan de compras calculado exitosamente")
            else:
                st.info("ℹ️ No se requieren compras adicionales según el análisis")
    
    with col2:
        if 'plan_compras' in st.session_state and not st.session_state.plan_compras.empty:
            st.subheader("📊 Resumen del Plan de Compras")
            
            plan_df = st.session_state.plan_compras
            
            # Métricas principales
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                total_insumos = len(plan_df['insumo'].unique())
                st.metric("Insumos a Comprar", total_insumos)
            
            with col_m2:
                costo_total = plan_df['costo_total'].sum()
                st.metric("Costo Total", f"${costo_total:,.0f}")
            
            with col_m3:
                tiempo_max = plan_df['tiempo_entrega'].max()
                st.metric("Tiempo Máx. Entrega", f"{tiempo_max} días")
            
            # Tabla del plan de compras
            st.subheader("📋 Detalle del Plan de Compras")
            
            # Ordenar por costo total descendente
            plan_display = plan_df.sort_values('costo_total', ascending=False)
            
            columnas_mostrar = ['insumo', 'producto', 'cantidad_comprar', 'costo_unitario', 
                              'costo_total', 'tiempo_entrega']
            
            st.dataframe(
                plan_display[columnas_mostrar].round(2),
                hide_index=True,
                height=300
            )
            
            # Gráfico de costos por insumo
            fig_costos = px.bar(
                plan_display.head(15), 
                x='insumo', 
                y='costo_total',
                title="Top 15 Insumos por Costo de Compra",
                color='tiempo_entrega',
                color_continuous_scale='Reds'
            )
            fig_costos.update_xaxes(tickangle=45)
            st.plotly_chart(fig_costos, use_container_width=True)
            
            # Análisis por producto
            st.subheader("📈 Análisis por Producto")
            
            analisis_producto = plan_df.groupby('producto').agg({
                'costo_total': 'sum',
                'insumo': 'count',
                'tiempo_entrega': 'max'
            }).reset_index()
            
            analisis_producto.columns = ['producto', 'costo_total', 'num_insumos', 'tiempo_max_entrega']
            
            fig_productos = px.scatter(
                analisis_producto,
                x='num_insumos',
                y='costo_total',
                size='tiempo_max_entrega',
                hover_name='producto',
                title="Análisis de Compras por Producto"
            )
            st.plotly_chart(fig_productos, use_container_width=True)
            
            # Cronograma de compras
            st.subheader("📅 Cronograma de Compras")
            
            # Calcular fecha de pedido considerando tiempo de entrega
            fecha_base = dt.date.today()
            plan_cronograma = plan_df.copy()
            plan_cronograma['fecha_pedido'] = fecha_base
            plan_cronograma['fecha_entrega'] = plan_cronograma.apply(
                lambda row: fecha_base + dt.timedelta(days=row['tiempo_entrega']), axis=1
            )
            
            # Agrupar por fecha de entrega
            cronograma_resumen = plan_cronograma.groupby('fecha_entrega').agg({
                'costo_total': 'sum',
                'insumo': 'count'
            }).reset_index()
            
            fig_cronograma = px.bar(
                cronograma_resumen,
                x='fecha_entrega',
                y='costo_total',
                title="Cronograma de Entregas y Costos"
            )
            st.plotly_chart(fig_cronograma, use_container_width=True)

elif menu == "⚙️ Optimización":
    st.header("Optimización de Inventarios")
    
    st.subheader("🎯 Análisis de Optimización")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Análisis ABC")
        
        # Análisis ABC por valor
        inventario_abc = st.session_state.inventario_df.copy()
        inventario_abc['valor_total'] = (inventario_abc['stock_planta'] + inventario_abc['stock_transito']) * inventario_abc['costo_unitario']
        inventario_abc = inventario_abc.sort_values('valor_total', ascending=False)
        
        # Calcular porcentajes acumulados
        inventario_abc['valor_acumulado'] = inventario_abc['valor_total'].cumsum()
        valor_total = inventario_abc['valor_total'].sum()
        inventario_abc['porcentaje_acumulado'] = (inventario_abc['valor_acumulado'] / valor_total) * 100
        
        # Clasificación ABC
        inventario_abc['clasificacion'] = inventario_abc['porcentaje_acumulado'].apply(
            lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C')
        )
        
        # Mostrar distribución ABC
        distribucion_abc = inventario_abc['clasificacion'].value_counts()
        fig_abc = px.pie(
            values=distribucion_abc.values,
            names=distribucion_abc.index,
            title="Distribución ABC de Insumos"
        )
        st.plotly_chart(fig_abc, use_container_width=True)
        
        # Estadísticas ABC
        st.markdown("#### Estadísticas por Categoría")
        for categoria in ['A', 'B', 'C']:
            datos_cat = inventario_abc[inventario_abc['clasificacion'] == categoria]
            if not datos_cat.empty:
                num_insumos = len(datos_cat)
                valor_promedio = datos_cat['valor_total'].mean()
                porcentaje_valor = (datos_cat['valor_total'].sum() / valor_total) * 100
                
                st.metric(
                    f"Categoría {categoria}",
                    f"{num_insumos} insumos",
                    f"{porcentaje_valor:.1f}% del valor"
                )
    
    with col2:
        st.markdown("### 🔄 Rotación de Inventarios")
        
        # Simular datos de consumo (en un sistema real vendría de datos históricos)
        np.random.seed(42)
        rotacion_data = []
        
        for _, insumo_row in st.session_state.inventario_df.iterrows():
            # Calcular consumo basado en BOM y demanda histórica
            insumo = insumo_row['insumo']
            
            # Encontrar productos que usan este insumo
            productos_insumo = st.session_state.bom_df[
                st.session_state.bom_df['insumo'] == insumo
            ]['producto'].unique()
            
            consumo_mensual = 0
            for producto in productos_insumo:
                # Demanda promedio del producto
                demanda_promedio = st.session_state.demanda_df[
                    st.session_state.demanda_df['producto'] == producto
                ]['demanda'].mean()
                
                # Cantidad de insumo por unidad
                cantidad_por_unidad = st.session_state.bom_df[
                    (st.session_state.bom_df['producto'] == producto) &
                    (st.session_state.bom_df['insumo'] == insumo)
                ]['cantidad_por_unidad'].iloc[0]
                
                consumo_mensual += demanda_promedio * cantidad_por_unidad
            
            # Calcular rotación
            stock_promedio = insumo_row['stock_planta']
            if stock_promedio > 0:
                rotacion_anual = (consumo_mensual * 12) / stock_promedio
            else:
                rotacion_anual = 0
            
            rotacion_data.append({
                'insumo': insumo,
                'consumo_mensual': consumo_mensual,
                'stock_promedio': stock_promedio,
                'rotacion_anual': rotacion_anual,
                'dias_cobertura': (stock_promedio / max(consumo_mensual/30, 0.1))
            })
        
        df_rotacion = pd.DataFrame(rotacion_data)
        
        # Gráfico de rotación vs días de cobertura
        fig_rotacion = px.scatter(
            df_rotacion,
            x='dias_cobertura',
            y='rotacion_anual',
            hover_name='insumo',
            title="Rotación vs Días de Cobertura",
            labels={
                'dias_cobertura': 'Días de Cobertura',
                'rotacion_anual': 'Rotación Anual'
            }
        )
        
        # Agregar líneas de referencia
        fig_rotacion.add_hline(y=12, line_dash="dash", line_color="red", 
                              annotation_text="Rotación Objetivo (12x/año)")
        fig_rotacion.add_vline(x=30, line_dash="dash", line_color="green", 
                              annotation_text="30 días cobertura")
        
        st.plotly_chart(fig_rotacion, use_container_width=True)
        
        # Insumos con baja rotación
        st.markdown("#### 🐌 Insumos con Baja Rotación")
        baja_rotacion = df_rotacion[df_rotacion['rotacion_anual'] < 6].sort_values('rotacion_anual')
        
        if not baja_rotacion.empty:
            st.dataframe(
                baja_rotacion[['insumo', 'rotacion_anual', 'dias_cobertura']].round(2),
                hide_index=True
            )
        else:
            st.success("✅ Todos los insumos tienen rotación adecuada")
    
    # Recomendaciones de optimización
    st.subheader("💡 Recomendaciones de Optimización")
    
    recomendaciones = []
    
    # Análisis de stock crítico
    stock_critico = st.session_state.inventario_df[
        st.session_state.inventario_df['stock_planta'] <= st.session_state.inventario_df['stock_minimo']
    ]
    
    if not stock_critico.empty:
        recomendaciones.append({
            'tipo': '⚠️ Stock Crítico',
            'descripcion': f"Hay {len(stock_critico)} insumos en stock crítico que requieren reposición inmediata",
            'prioridad': 'Alta',
            'accion': 'Generar órdenes de compra urgentes'
        })
    
    # Análisis de exceso de inventario
    if 'df_rotacion' in locals():
        exceso_stock = df_rotacion[df_rotacion['dias_cobertura'] > 90]
        if not exceso_stock.empty:
            recomendaciones.append({
                'tipo': '📦 Exceso de Inventario',
                'descripcion': f"Hay {len(exceso_stock)} insumos con más de 90 días de cobertura",
                'prioridad': 'Media',
                'accion': 'Reducir órdenes de compra para estos insumos'
            })
    
    # Análisis de costos altos
    inventario_caro = st.session_state.inventario_df.nlargest(5, 'costo_unitario')
    if not inventario_caro.empty:
        recomendaciones.append({
            'tipo': '💰 Insumos de Alto Costo',
            'descripcion': f"Los 5 insumos más caros representan un alto valor en inventario",
            'prioridad': 'Media',
            'accion': 'Evaluar proveedores alternativos y negociar precios'
        })
    
    # Análisis de tiempo de entrega
    tiempo_largo = st.session_state.inventario_df[st.session_state.inventario_df['tiempo_entrega_dias'] > 20]
    if not tiempo_largo.empty:
        recomendaciones.append({
            'tipo': '⏰ Tiempos de Entrega Largos',
            'descripcion': f"Hay {len(tiempo_largo)} insumos con tiempo de entrega superior a 20 días",
            'prioridad': 'Media',
            'accion': 'Aumentar stock de seguridad o buscar proveedores locales'
        })
    
    if recomendaciones:
        df_recomendaciones = pd.DataFrame(recomendaciones)
        
        # Mostrar por prioridad
        for prioridad in ['Alta', 'Media', 'Baja']:
            recs_prioridad = df_recomendaciones[df_recomendaciones['prioridad'] == prioridad]
            if not recs_prioridad.empty:
                st.markdown(f"#### Prioridad {prioridad}")
                for _, rec in recs_prioridad.iterrows():
                    with st.expander(rec['tipo']):
                        st.write(f"**Descripción:** {rec['descripcion']}")
                        st.write(f"**Acción recomendada:** {rec['accion']}")
    else:
        st.success("✅ El inventario está optimizado según los parámetros analizados")
    
    # Simulador de escenarios
    st.subheader("🎲 Simulador de Escenarios")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Configuración del Escenario")
        
        escenario = st.selectbox(
            "Tipo de escenario:",
            ["Incremento de demanda", "Reducción de demanda", "Problemas de suministro", "Cambio de precios"]
        )
        
        if escenario == "Incremento de demanda":
            factor_cambio = st.slider("Factor de incremento:", 1.1, 2.0, 1.3, 0.1)
        elif escenario == "Reducción de demanda":
            factor_cambio = st.slider("Factor de reducción:", 0.5, 0.9, 0.7, 0.1)
        elif escenario == "Problemas de suministro":
            factor_cambio = st.slider("Incremento en tiempo de entrega:", 1.5, 3.0, 2.0, 0.1)
        else:  # Cambio de precios
            factor_cambio = st.slider("Factor de cambio en precios:", 0.8, 1.5, 1.2, 0.1)
        
        simular = st.button("🔄 Simular Escenario")
    
    with col2:
        if simular:
            st.markdown("#### Resultados de la Simulación")
            
            if escenario == "Incremento de demanda":
                st.write(f"📈 **Escenario:** Demanda aumenta {(factor_cambio-1)*100:.0f}%")
                
                # Calcular impacto en necesidades de compra
                if 'plan_compras' in st.session_state:
                    plan_original = st.session_state.plan_compras['costo_total'].sum()
                    plan_nuevo = plan_original * factor_cambio
                    diferencia = plan_nuevo - plan_original
                    
                    st.metric("Costo de Compras Original", f"${plan_original:,.0f}")
                    st.metric("Costo de Compras Nuevo", f"${plan_nuevo:,.0f}", f"+${diferencia:,.0f}")
                    
                    st.write("**Impactos:**")
                    st.write("- Necesidad de aumentar órdenes de compra")
                    st.write("- Posible necesidad de proveedores adicionales")
                    st.write("- Riesgo de desabastecimiento si no se ajusta rápidamente")
                
            elif escenario == "Reducción de demanda":
                st.write(f"📉 **Escenario:** Demanda disminuye {(1-factor_cambio)*100:.0f}%")
                
                if 'plan_compras' in st.session_state:
                    plan_original = st.session_state.plan_compras['costo_total'].sum()
                    plan_nuevo = plan_original * factor_cambio
                    ahorro = plan_original - plan_nuevo
                    
                    st.metric("Costo de Compras Original", f"${plan_original:,.0f}")
                    st.metric("Costo de Compras Nuevo", f"${plan_nuevo:,.0f}", f"-${ahorro:,.0f}")
                    
                    st.write("**Impactos:**")
                    st.write("- Reducción en órdenes de compra")
                    st.write("- Posible exceso de inventario")
                    st.write("- Oportunidad de optimizar costos de almacenamiento")
            
            elif escenario == "Problemas de suministro":
                st.write(f"⏰ **Escenario:** Tiempos de entrega aumentan {(factor_cambio-1)*100:.0f}%")
                
                tiempo_promedio = st.session_state.inventario_df['tiempo_entrega_dias'].mean()
                nuevo_tiempo = tiempo_promedio * factor_cambio
                
                st.metric("Tiempo Entrega Promedio Original", f"{tiempo_promedio:.1f} días")
                st.metric("Tiempo Entrega Promedio Nuevo", f"{nuevo_tiempo:.1f} días", 
                         f"+{nuevo_tiempo-tiempo_promedio:.1f} días")
                
                st.write("**Impactos:**")
                st.write("- Necesidad de aumentar stock de seguridad")
                st.write("- Mayor capital inmovilizado en inventario")
                st.write("- Riesgo de desabastecimiento durante transición")
            
            else:  # Cambio de precios
                if factor_cambio > 1:
                    st.write(f"💰 **Escenario:** Precios aumentan {(factor_cambio-1)*100:.0f}%")
                else:
                    st.write(f"💰 **Escenario:** Precios disminuyen {(1-factor_cambio)*100:.0f}%")
                
                valor_actual = (st.session_state.inventario_df['stock_planta'] * 
                              st.session_state.inventario_df['costo_unitario']).sum()
                valor_nuevo = valor_actual * factor_cambio
                diferencia = valor_nuevo - valor_actual
                
                st.metric("Valor Inventario Original", f"${valor_actual:,.0f}")
                st.metric("Valor Inventario Nuevo", f"${valor_nuevo:,.0f}", f"${diferencia:,.0f}")
                
                st.write("**Impactos:**")
                if factor_cambio > 1:
                    st.write("- Mayor costo de reposición")
                    st.write("- Considerar compras anticipadas")
                    st.write("- Evaluar impacto en rentabilidad")
                else:
                    st.write("- Menor costo de reposición")
                    st.write("- Oportunidad de aumentar stock")
                    st.write("- Mejora en márgenes de contribución")

# Pie de página
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <small>Sistema de Pronóstico de Demanda y Gestión de Inventarios | Desarrollado SAINECO soportado en IA</small>
    </div>
    """, 
    
    unsafe_allow_html=True
    
)