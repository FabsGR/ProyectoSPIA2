% Cargar el conjunto de datos
data = readtable('zoo.csv'); % Asegúrate de proporcionar la ruta correcta al archivo CSV

% Dividir el conjunto de datos en características (X) y etiquetas (y)
X = data{:, 2:end-1}; % Excluir la primera columna (animal_name) y la última columna (class_type)
y = data{:, end};     % La última columna es la etiqueta (class_type)

% Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
rng(7845); % Establecer la semilla para la reproducibilidad
cv = cvpartition(length(y), 'Holdout', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Normalizar las características (opcional pero recomendado para algunos algoritmos)
X_train = normalize(X_train);
X_test = normalize(X_test);

% Regresión Logística Multiclase
mdl_multiclass = fitmnr(X_train, y_train);
y_pred_multiclass = predict(mdl_multiclass, X_test);


% K-Vecinos Cercanos
mdl_knn = fitcknn(X_train, y_train, 'NumNeighbors', 5);
y_pred_knn = predict(mdl_knn, X_test);


% Máquinas de Vectores de Soporte
mdl_svm = fitcecoc(X_train, y_train);
y_pred_svm = predict(mdl_svm, X_test);


% Naive Bayes
% Encuentra las columnas con varianza extremadamente baja
low_var_cols = (var(X_train) < 1e-10);

% Elimina las columnas con varianza extremadamente baja
X_train_nb = X_train(:, ~low_var_cols);
X_test_nb = X_test(:, ~low_var_cols);

% Entrena el modelo Naive Bayes nuevamente
mdl_nb = fitcnb(X_train_nb, y_train, 'DistributionNames', 'kernel');
y_pred_nb = predict(mdl_nb, X_test_nb);


% Cargar la función Evaluate
evalMetrics = @(actual, predicted) Evaluate(actual, predicted);

% Calcular las métricas de rendimiento para cada modelo
eval_logit = evalMetrics(y_test, y_pred_multiclass);
eval_knn = evalMetrics(y_test, y_pred_knn);
eval_svm = evalMetrics(y_test, y_pred_svm);
eval_nb = evalMetrics(y_test, y_pred_nb);

% Imprimir las métricas de rendimiento
fprintf('Logistic Regression Metrics:     Accuracy=%.2f, Sensitivity=%.2f, Specificity=%.2f, Precision=%.2f, Recall=%.2f, F1=%.2f, G-mean=%.2f\n', eval_logit);
fprintf('K-Nearest Neighbors Metrics:     Accuracy=%.2f, Sensitivity=%.2f, Specificity=%.2f, Precision=%.2f, Recall=%.2f, F1=%.2f, G-mean=%.2f\n', eval_knn);
fprintf('Support Vector Machines Metrics: Accuracy=%.2f, Sensitivity=%.2f, Specificity=%.2f, Precision=%.2f, Recall=%.2f, F1=%.2f, G-mean=%.2f\n', eval_svm);
fprintf('Naive Bayes Metrics:             Accuracy=%.2f, Sensitivity=%.2f, Specificity=%.2f, Precision=%.2f, Recall=%.2f, F1=%.2f, G-mean=%.2f\n\n', eval_nb);


% Crear una matriz con las métricas de rendimiento
metrics_matrix = [eval_logit; eval_knn; eval_svm; eval_nb];

% Crear una tabla con las métricas de rendimiento
metrics_table = array2table(metrics_matrix, 'VariableNames', {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1', 'GMean'}, ...
                  'RowNames', {'Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machines', 'Naive Bayes'});

% Imprimir la tabla
disp(metrics_table)











