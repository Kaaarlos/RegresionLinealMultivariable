clear all;
close all;

datos = load("datos.txt");

x = datos(:, 1:2);  
y = datos(:, 3); 
m = numel(y);

plot3(x(:,1),x(:,2),y, 'ok', 'MarkerFaceColor', 'y');
xlabel('x1');
ylabel('x2');
zlabel('y');
hold on

%Normalizar 
[~,n]=size(x);
x_norm=zeros(m,n);
mu=mean(x);
sigma=std(x,1); %desvisacion estandar

for i=1:n
    x_norm(:,i)=(x(:,i)-mu(i))/sigma(i);
end
figure(2)
plot3(x_norm(:,1),x_norm(:,2),y, 'ok', 'MarkerFaceColor', 'y');
xlabel('x1');
ylabel('x2');
zlabel('y');

x=x_norm;
x=[ones(m,1),x];

n=n+1;
a=zeros(n,1);
beta = .8; 

iterMax = 600;
iter = 1; 

for i=1:m
    h(i,1) = a'*x(i,:)'; 
end

J = (1 / (2 * m )) * sum((h - y).^2); 

while(iter < iterMax)
 convergencia(iter) = J;
 for j=1:n
     a(j)=a(j)-(beta*((1/m)*sum((h-y).*x(:,j))));
 end

 for i=1:m
    h(i,1) = a'*x(i,:)'; 
 end
 J = (1 / (2 * m )) * sum((h - y).^2);
 iter = iter + 1;
end

figure(1)
Prueba1 = [3000, 4, 539900];
PruebaNorm = (Prueba1(:, 1:2) - mu)./sigma; 
PruebaNorm = [1 PruebaNorm];
hDatoPrueba1 = a' * PruebaNorm'; 
plot3(PruebaNorm(2),PruebaNorm(3), hDatoPrueba1, 'ok','MarkerFaceColor','r');

Prueba2 = [1985, 4, 299900]; 
PruebaNorm = (Prueba2(:, 1:2) - mu)./sigma;
PruebaNorm = [1 PruebaNorm]; 
hDatoPrueba2 = a' * PruebaNorm'; 
plot3(datoPruebaNorm(2), PruebaNorm(3), hDatoPrueba2, 'ok','MarkerFaceColor','b');

Prueba3 = [1534, 3, 314900];
PruebaNorm = (Prueba3(:, 1:2) - mu)./sigma; 
PruebaNorm = [1 PruebaNorm]; 
hDatoPrueba3 = a' * PruebaNorm'; 
plot3(PruebaNorm(2), PruebaNorm(3), hDatoPrueba3, 'ok','MarkerFaceColor','g');

disp('Resultados: ');
fprintf('J = %.4f\n', J);
fprintf('a0 = %.4f\n', a(1));
fprintf('a1 = %.4f\n', a(2));
fprintf('a2 = %.4f\n', a(3));
fprintf(['Dato de prueba 1 x1= %d x2= %d Salida correcta y= %d' ...
' Predicción h= %.4f \n'], datoPrueba1(1), datoPrueba1(2), ...
datoPrueba1(3), hDatoPrueba1);
fprintf(['Dato de prueba 2 x1= %d x2= %d Salida correcta y= %d' ...
' Predicción h= %.4f \n'], datoPrueba2(1), datoPrueba2(2), ...
datoPrueba2(3), hDatoPrueba2);
fprintf(['Dato de prueba 3 x1= %d x2= %d Salida correcta y= %d' ...
' Predicción h= %.4f \n'], datoPrueba3(1), datoPrueba3(2), ...
datoPrueba3(3), hDatoPrueba3);



plot(convergencia);
