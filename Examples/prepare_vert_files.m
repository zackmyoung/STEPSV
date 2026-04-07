function []= prepare_vert_files(stlst)

%% NOTE! This example does not prepare NTAOL corrections for the time series or any additional cleaning. 
%% Examples are provided to illustrate the workflow only.
%% Results without NTAOL corrections are expected to exhibit slight differences to those corrected. 
%% NTAOL corrections can be obtained from GFZ (Dill and Dobslaw 2012). 

for a =1:length(stlst(:,1))
    st=stlst(a,:);
    setenv('st',st)
    !echo $st
    !wget -P tenv3/ https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/`echo $st`.tenv3  

    d=importdata(['tenv3/' st '.tenv3']);
    T=d.data(:,1);
    U=d.data(:,[11 ])*1000;
    Sig=d.data(:,[15])*1000;
    ant=d.data(:,[12])*1000; 


    i=ones(length(T),1);
    sin2pi=sin((2*pi*T));
    cos2pi=cos((2*pi*T));
    sin4pi=sin((4*pi*T));
    cos4pi=cos((4*pi*T));

    A=[i T sin2pi cos2pi sin4pi  cos4pi ]; 

    W=diag((1./(Sig(:,1).^2))); 
    a1=A(1,:); 
    Ag=inv(transpose(A)*W*A)*transpose(A)*W;
    mU=Ag*U;
    % lineE=A(:,1:2)*mE(1:2,1); 
    lineU=A*mU; 

    dat_detrend=[T U-lineU Sig ant];

%     figure
%     scatter(T, U,'k')
%     hold on
%     scatter(T,U-lineU,'r')


    fid = fopen(['GPS_data/' st '.data'],'w');
    fprintf(fid,'%.6f %.6f %.6f %.0f\n',dat_detrend');
    fclose(fid);
end