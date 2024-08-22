% running problem one. Primal problem. Prepared to run off-line
%lmax=id1; Nmax=id2; M = id3;
% [6,8,200,200,8],[6,8,64,64,20],[3,10,64,64,20],[3,10,64,64,10]
% [3,8,200,200,8],[3,8,200,200,20],[4,8,200,200,8],[5,8,200,200,8]

lmax=8; Nmax=14; M=200; Lambda=200; Nmax2=14;

xiV=pi/M*((1:Lambda)-1/2);    
sV =4./cos(xiV/2).^2; 
fgV=kron(ones(1,lmax),(pi/4)*sqrt((sV-4)./sV));
dl0 = kron([1 zeros(1,lmax-1)],ones(1,Lambda));
A=2*dl0';

Mflsp=lmax*Lambda; 
MfnS=(Nmax+1)*(Nmax+2)/2;
pole = 'pole';

% M is the number of points on the half circle
dataKFile=sprintf('SmQCD_K_data_Psp_%s0_lmax%d_Na%d_M%d_Lambda%d_Na2%d_A.mat',pole,lmax,Nmax,M,Lambda,Nmax2);
dataWFile=sprintf('SmQCD_W_data_Psp_%s0_lmax%d_Na%d_M%d_Lambda%d_Na2%d_A.mat',pole,lmax,Nmax,M,Lambda,Nmax2);

dataOutFile=sprintf('SmQCD_sp_Fp_%s_norm2_lmax%d_Na%d_M%d_Lambda%d_Na2%d_A_out.dat',pole,lmax,Nmax,M,Lambda,Nmax2);
load(dataKFile);
load(dataWFile);


fileID = fopen(dataOutFile,'w');

%alphav=[0:0.01:2*pi];
alphav=[0:0.01:2*pi];
%mreg = [-2:0.1:6];
mreg=[2 3 4 5]
%mreg = [-2:0.5:-2];
tv = zeros(size(mreg.'*alphav));
fdv = zeros(size(mreg.'*alphav));
t00v = zeros(size(mreg.'*alphav));
t20v = zeros(size(mreg.'*alphav));

count_alpha=1;


for alpha =alphav
    count_reg=1;
for Mreg = 10.^mreg
%     Mreg=100;    

     cvx_begin quiet;
     cvx_solver Mosek_2;
%     cvx_precision high;

     variable f0;                                   % constant
     variable sigma1(Nmax2+1);                      % single dispersion
     variable rho1(MfnS);                           % double dispersion
     variable t;

     Imh = fgV'.*(     Alni*sigma1+Alnmi*rho1);     % partial waves
     Reh = fgV'.*(A*f0+Alnr*sigma1+Alnmr*rho1);     %
     Reh.*Reh+Imh.*Imh <= 2*Imh;                    % unitarity bound
     
     norm(rho1,2)<=Mreg;
     
%     sigma1(1)==0;                                 % remove threshold pole
%     rho1(1:Nmax+1)==0;

%     t*cos(alpha) == pi/4*(f0+wsigma*sigma1+wrho*rho1);        % lambda00 functional
%     0.2+t*sin(alpha) == pi/4*(w20s*sigma1+w20r*rho1);       % lambda20 functional

    t1 = pi/4*(f0+wsigma*sigma1+wrho*rho1);        % lambda00 functional
    t2 = pi/4*(w20s*sigma1+w20r*rho1);             % lambda20 functional

%     t1 == t*cos(alpha);
%     t2 == t*sin(alpha)+0.2;

%     t==t2;
     t==t1*cos(alpha)+t2*sin(alpha);

     maximize(t);                                   % maximization

  cvx_end
  
  
  if ~strcmp(cvx_status,'Failed') 
       tv(count_reg,count_alpha) = t;
       t00v(count_reg,count_alpha) = t1;
       t20v(count_reg,count_alpha) = t2;
       fdv(count_reg,count_alpha)= norm(rho1(:),2)/Mreg;
  end
  
  
%  count=count+1;
  
  % print on screen and file
  fprintf('[%5.3f\t %10.3f\t %4.20f\t %10.3f\t %s],\n',alpha,Mreg,t,norm(rho1(:),2),cvx_status);
  fprintf(fileID,'%4.20f %4.20f %4.20f \n',Mreg,t,norm(rho1(:),2));
  % print partial waves
  %for l = 0:4
  %      fprintf(fileID,' %4.20f',Ref(l*(M+1)+(1:M+1))); fprintf(fileID,'\n');
  %      fprintf(fileID,' %4.20f',Imf(l*(M+1)+(1:M+1))); fprintf(fileID,'\n');
  %end


 count_reg=count_reg+1;
end
 count_alpha=count_alpha+1;
end

 fclose(fileID);

%figure
%plot(mreg,tv(:,1),'mo',mreg,fdv,'ro');

%figure
%plot(mreg,2.6613*ones(size(mreg)),'g');
%leaf=imread('Leaf.png');
%imshow(leaf);
hold on
%plot(mreg,tv,'mo',mreg,fdv,'ro');
%plot(t00v(1,:),t20v(1,:),'mo');
%plot(t00v(2,:),t20v(2,:),'ro');
%plot(t00v(3,:),t20v(3,:),'bo');
%plot(t00v(4,:),t20v(4,:),'ko');
color='bxmxrx';
for j=1:count_reg-1
    j1=2*rem(j,fix(length(color)/2))+1;
    plot(t00v(j,:),t20v(j,:),color(j1:j1+1));
end
plot([2.6613 2.6613],[0.0 0.9],'r')
plot([-7.2 -7.2],[0 0.9],'r')
plot([-7.2 2.6613],[0.9 0.9],'r')
plot([-7.2 2.6613],[0 0],'r')
plot([0 0],[0.0 0.9],'k')
plot([-7.2 2.6613],[0.4 0.4],'k')
xlabel('c0')
xlim([-8 3])
ylabel('c2')
ylim([-0.1 1.0])
title('Leaf')
