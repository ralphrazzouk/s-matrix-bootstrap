% PRIMAL PROBLEM 2.0
% Uses rescaling, intermediate points, and full subtraction 

% Running problem 2 using interpolation points.
% Prepared to run offline (remove plots)

% M is the number of points on the half circle
% saXXX is the subtractions point s0 = XXX

% st1 = ["-1p00", "0p00", "1p33", "2p00", "3p00", "3p80"];
% st1_python = ["-1", "0", "1p33", "2", "3", "3p8"];

st1 = ["2p00"];
ct1 = ["ko", "go", "ko", "co", "yo"];
Mv = [40];
lmaxv = [20];

mreg = [8];
nfv = [1];
nv = [50];

% mreg = [-4:0.2:10];
% mreg = [2 4 6 8];
% mreg = [-4 0 2 4 7 9];  % regulator
% nfv = [1 7 9 13];       % ntheta
% nv = [4:2:20];          % ncoeffs Requires having M from previous run
% tv = zeros(length(st1), length(mreg));
% alphav = [0:0.1:2*pi];
% tv = zeros(length(st1), length(mreg) * length(nv) * length(nfv));

alphav = [0:0.001:2*pi];
tv  = zeros(length(st1), length(mreg), length(alphav));
c0v = zeros(length(st1), length(mreg), length(alphav));
c2v = zeros(length(st1), length(mreg), length(alphav));
fdv = zeros(size(mreg));

for jcount = 1:length(st1)
    M = Mv(jcount);
    lmax = lmaxv(jcount);
    
    dataKFile = sprintf('primal_2.0_points_data_P2_nopole_sa%s_lmax%d_M%d.dat', st1(jcount), lmax, M);
    dataOutFile = sprintf('primal_2.0_points_data_P2_nopole_sa%s_lmax%d_M%d_ncoeff%d_outB.datx', st1(jcount), lmax, M, max(nv));

    Mfl = (lmax + 1) * M;
    Mfn = M * (M + 1) / 2;
    Mfn2 = (M + 1) * (M + 2) / 2;
    
    fileID = fopen(dataKFile, 'r');
    xiv = fscanf(fileID, '%f', [M]);

    % REGULAR
    a0h  = fscanf(fileID, '%f', 1);
    b0h  = fscanf(fileID, '%f', 1);
    h0R  = fscanf(fileID, '%f', [Mfl]);
    a1h  = fscanf(fileID, '%f', [M]);
    b1h  = fscanf(fileID, '%f', [M]);
    h1R  = fscanf(fileID, '%f', [Mfl M]);
    h1I  = fscanf(fileID, '%f', [Mfl M]);
    a2h  = fscanf(fileID, '%f', [Mfn]);
    b2h  = fscanf(fileID, '%f', [Mfn]);
    h2R  = fscanf(fileID, '%f', [Mfl Mfn]);
    h2I  = fscanf(fileID, '%f', [Mfl Mfn]);

    % RESCALED
    h0RA  = fscanf(fileID, '%f', [Mfl]);
    h1RA  = fscanf(fileID, '%f', [Mfl M]);
    h1IA  = fscanf(fileID, '%f', [Mfl M]);
    h2RA  = fscanf(fileID, '%f', [Mfl Mfn]);
    h2IA  = fscanf(fileID, '%f', [Mfl Mfn]);
    h1IB  = fscanf(fileID, '%f', [Mfl M]);
    h2IB  = fscanf(fileID, '%f', [Mfl Mfn]);
    Lambdav = fscanf(fileID, '%f', [Mfl]);
    
    fclose(fileID);

    % REORDER AND RECONDITION LINEAR CONSTRAINT MATRIX
    A = [[h0RA h1RA h2RA]; [zeros(Mfl, 1) h1IA h2IA]; [zeros(Mfl, 1) h1IB h2IB]]; % with rescaling 
     
    [Q, R, P] = qr(A', 'vector');
    ap = [a0h a1h' a2h'] * Q;
    bp = [b0h b1h' b2h'] * Q;
    
    invP(P) = 1:length(P);
    indRA = invP(1:Mfl);
    indIA = invP(Mfl+1:2*Mfl);
    indIB = invP(2*Mfl+1:3*Mfl);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fileID = fopen(dataOutFile, 'w');
    
    sv = 4./cos(xiv/2).^2;
    count_alpha = 1;
    
    for alpha = alphav
        count=1;
        for nf = nfv
            for n=nv  
                for Mreg = 10.^mreg
                    cvx_begin quiet;
                    cvx_solver Mosek_2; 
                    % cvx_precision high;
    
                    variable v(Mfn2);
                    variable t;
                  
                    % REDUCED VARIABLES
                    v(n * (M + 1) + 1:end) == 0;
                    hf1 = R' * v;
                    
                    hRA = hf1(indRA);
                    hIA = hf1(indIA);
                    hIB = hf1(indIB);
                    
                    % UNITARITY
                    norms([hRA hIA], 2, 2) <= sqrt(2*hIB);
                
                    % REGULARIZATION
                    rhon = norm(v(nf * (M + 1) + 1:end), 32);
                    rhon <= Mreg;
                    
                    % FUNCTIONAL
                    c0 = (pi/4) * ap * v;
                    c2 = (pi/4) * bp * v;
                
                    t == c0 * cos(alpha) + c2*sin(alpha);
                
                    maximize(t);

                    cvx_end
    
                    tv(jcount, count, count_alpha) = t;
                    if ~strcmp(cvx_status, 'Failed') 
                        c0v(jcount, count, count_alpha) = c0;
                        c2v(jcount, count, count_alpha) = c2;
                    end

                    fdv(count) = rhon;
        
                    fprintf(      '[%5.3f\t %10.3f\t %4.20f\t %5.3f\t %5.3f\t %10.3f\t %s],\n',alpha,Mreg,t,c0,c2,fdv(count),cvx_status);
                    fprintf(fileID,'%5.3f\t %10.3f\t %4.20f\t %5.3f\t %5.3f\t %10.3f\t %s\n'  ,alpha,Mreg,t,c0,c2,fdv(count),cvx_status);
                    
                    count=count+1;
     
                end % REGULATOR mreg
            end % NUMBER OF COEFFICIENTS ncoeff
        end % NUMBER OF REGULATED COMPONENTS ntheta

    count_alpha = count_alpha + 1;
    end % ALPHA ANGLE IN c0 c2-plane

    fclose(fileID);
end % SUBTRACTION POINT


% FIGURE
hold on
color = 'morobo';

for ja = 1:length(st1)
    for j = 1:count - 1
        j1 = 2*rem(j, fix(length(color)/2)) - 1;
        plot(squeeze(c0v(ja, j, :)), squeeze(c2v(ja, j, :)), color(j1:j1 + 1));
    end
end

plot([2.6613 2.6613], [0.0 0.9], 'r')
plot([-7.2 -7.2], [0 0.9], 'r')
plot([-7.2 2.6613], [0.9 0.9], 'r')
plot([-7.2 2.6613], [0 0], 'r')
plot([0 0], [0.0 0.9], 'k')
plot([-7.2 2.6613], [0.4 0.4], 'k')
xlabel('c0')
xlim([-8 3])
ylabel('c2')
ylim([-0.1 1.0])
title('Leaf')
