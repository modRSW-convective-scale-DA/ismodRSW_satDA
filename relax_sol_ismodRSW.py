import numpy as np

def U_relax_1(Neq,Nk,L,V,xc,U):

    U_rel = np.zeros((Neq,Nk))

    Lj = 0.1*L
    f1 = V*(1+np.tanh(4*(xc-0.2*L)/Lj + 2))*(1-np.tanh(4*(xc-0.2*L)/Lj - 2))/4
    f2 = V*(1+np.tanh(4*(xc-0.4*L)/Lj + 2))*(1-np.tanh(4*(xc-0.4*L)/Lj - 2))/4
    f3 = V*(1+np.tanh(4*(xc-0.6*L)/Lj + 2))*(1-np.tanh(4*(xc-0.6*L)/Lj - 2))/4
    f4 = V*(1+np.tanh(4*(xc-0.8*L)/Lj + 2))*(1-np.tanh(4*(xc-0.8*L)/Lj - 2))/4
    ic3 = f1-f2+f3-f4
    U_rel[2,:] = ic3

    return U_rel

def U_relax_2(Neq,Nk,L,V,xc,U):

    U_rel = np.zeros((Neq,Nk))
    U_rel[2,:] = V*np.ones(Nk)

    return U_rel

def U_relax_3(Neq,Nk,L,V,xc,U):

    U_rel = np.zeros((Neq,Nk))

    Lj = 0.1*L
    f1 = V*(1+np.tanh(4*(xc-0.2*L)/Lj + 2))*(1-np.tanh(4*(xc-0.2*L)/Lj - 2))/4
    f2 = V*(1+np.tanh(4*(xc-0.4*L)/Lj + 2))*(1-np.tanh(4*(xc-0.4*L)/Lj - 2))/4
    f3 = V*(1+np.tanh(4*(xc-0.6*L)/Lj + 2))*(1-np.tanh(4*(xc-0.6*L)/Lj - 2))/4
    f4 = V*(1+np.tanh(4*(xc-0.8*L)/Lj + 2))*(1-np.tanh(4*(xc-0.8*L)/Lj - 2))/4
    ic3 = f1-f2+f3-f4
    U_rel[2,:] = (1+ic3)

    return U_rel

def U_relax_4(neq,nk,l,v,xc,u):

    u_rel = np.zeros((neq,nk))

    lj = 0.1*l
    f1 = v*(1+np.tanh(4*(xc-0.2*l)/lj + 2))*(1-np.tanh(4*(xc-0.2*l)/lj - 2))/4
    f2 = v*(1+np.tanh(4*(xc-0.4*l)/lj + 2))*(1-np.tanh(4*(xc-0.4*l)/lj - 2))/4
    f3 = v*(1+np.tanh(4*(xc-0.6*l)/lj + 2))*(1-np.tanh(4*(xc-0.6*l)/lj - 2))/4
    f4 = v*(1+np.tanh(4*(xc-0.8*l)/lj + 2))*(1-np.tanh(4*(xc-0.8*l)/lj - 2))/4
    ic3 = f1+f2+f3+f4
    u_rel[2,:] = ic3

    return u_rel

def U_relax_5(neq,nk,l,v,xc,u):

    u_rel = np.zeros((neq,nk))

    lj = 0.05*l
    f2 = v*(1+np.tanh(4*(xc-0.2*l)/lj + 2))*(1-np.tanh(4*(xc-0.2*l)/lj - 2))/4
    f3 = v*(1+np.tanh(4*(xc-0.3*l)/lj + 2))*(1-np.tanh(4*(xc-0.3*l)/lj - 2))/4
    f4 = v*(1+np.tanh(4*(xc-0.4*l)/lj + 2))*(1-np.tanh(4*(xc-0.4*l)/lj - 2))/4
    f6 = v*(1+np.tanh(4*(xc-0.6*l)/lj + 2))*(1-np.tanh(4*(xc-0.6*l)/lj - 2))/4
    f7 = v*(1+np.tanh(4*(xc-0.7*l)/lj + 2))*(1-np.tanh(4*(xc-0.7*l)/lj - 2))/4
    f8 = v*(1+np.tanh(4*(xc-0.8*l)/lj + 2))*(1-np.tanh(4*(xc-0.8*l)/lj - 2))/4
    ic3 = f2+f3+f4+f6+f7+f8
    u_rel[2,:] = ic3

    return u_rel

def U_relax_6(neq,nk,l,v,xc,u):

    u_rel = np.zeros((neq,nk))

    lj = 0.05*l
    f2 = v*(1+np.tanh(4*(xc-0.2*l)/lj + 2))*(1-np.tanh(4*(xc-0.2*l)/lj - 2))/4
    f3 = v*(1+np.tanh(4*(xc-0.3*l)/lj + 2))*(1-np.tanh(4*(xc-0.3*l)/lj - 2))/4
    f4 = v*(1+np.tanh(4*(xc-0.4*l)/lj + 2))*(1-np.tanh(4*(xc-0.4*l)/lj - 2))/4
    f6 = v*(1+np.tanh(4*(xc-0.6*l)/lj + 2))*(1-np.tanh(4*(xc-0.6*l)/lj - 2))/4
    f7 = v*(1+np.tanh(4*(xc-0.7*l)/lj + 2))*(1-np.tanh(4*(xc-0.7*l)/lj - 2))/4
    f8 = v*(1+np.tanh(4*(xc-0.8*l)/lj + 2))*(1-np.tanh(4*(xc-0.8*l)/lj - 2))/4
    ic3 = -f2-f3+f4-f6+f7+f8
    u_rel[2,:] = ic3

    return u_rel

def U_relax_7(neq,nk,l,v,xc,u):

    u_rel = np.zeros((neq,nk))

    lj = 0.05*l
    f1 = v*(1+np.tanh(4*(xc-0.1*l)/lj + 2))*(1-np.tanh(4*(xc-0.1*l)/lj - 2))/4
    f2 = v*(1+np.tanh(4*(xc-0.2*l)/lj + 2))*(1-np.tanh(4*(xc-0.2*l)/lj - 2))/4
    f3 = v*(1+np.tanh(4*(xc-0.3*l)/lj + 2))*(1-np.tanh(4*(xc-0.3*l)/lj - 2))/4
    f4 = v*(1+np.tanh(4*(xc-0.4*l)/lj + 2))*(1-np.tanh(4*(xc-0.4*l)/lj - 2))/4
    f6 = v*(1+np.tanh(4*(xc-0.6*l)/lj + 2))*(1-np.tanh(4*(xc-0.6*l)/lj - 2))/4
    f7 = v*(1+np.tanh(4*(xc-0.7*l)/lj + 2))*(1-np.tanh(4*(xc-0.7*l)/lj - 2))/4
    f8 = v*(1+np.tanh(4*(xc-0.8*l)/lj + 2))*(1-np.tanh(4*(xc-0.8*l)/lj - 2))/4
    f9 = v*(1+np.tanh(4*(xc-0.9*l)/lj + 2))*(1-np.tanh(4*(xc-0.9*l)/lj - 2))/4
    ic3 = f1-f3+f4-f6+f7-f9
    u_rel[2,:] = ic3

    return u_rel

def U_relax_8(neq,nk,l,v,xc,u):

    u_rel = np.zeros((neq,nk))

    lj = 0.1*l
    f1 = v*(1+np.tanh(4*(xc-0.5*l)/lj + 2))*(1-np.tanh(4*(xc-0.5*l)/lj - 2))/4
    ic3 = f1
    u_rel[2,:] = ic3

    return u_rel

def U_relax_9(neq,nk,l,v,xc,u):

    u_rel = np.zeros((neq,nk))

    lj = 0.1*l
    f1 = v*(1+np.tanh(4*(xc-0.25*l)/lj + 2))*(1-np.tanh(4*(xc-0.25*l)/lj - 2))/4
    f2 = v*(1+np.tanh(4*(xc-0.75*l)/lj + 2))*(1-np.tanh(4*(xc-0.75*l)/lj - 2))/4
    ic3 = f1+f2
    u_rel[2,:] = ic3

    return u_rel

def U_relax_10(neq,nk,l,v,xc,u): # double positive jet -- top-hat

    u_rel = np.zeros((neq,nk))
 
    a1 = 0.1
    a2 = 0.1
    ic3 = v*(np.heaviside(a1-abs(xc-l/4),1.)+np.heaviside(a2-abs(xc-3*l/4),1.)) 

    u_rel[2,:] = ic3

    return u_rel

def U_relax_11(neq,nk,l,v,xc,u): # double positive jet -- smoothed

    u_rel = np.zeros((neq,nk))
 
    a1 = 0.1
    a2 = 0.1
    lj = 0.05*l
    ic3 = v*(1+np.tanh(4*(xc-(l/4-l/16))/lj + 2))*(1-np.tanh(4*(xc-(l/4+l/16))/lj - 2))/4
    ic4 = v*(1+np.tanh(4*(xc-(3*l/4-l/16))/lj + 2))*(1-np.tanh(4*(xc-(3*l/4+l/16))/lj - 2))/4   

    u_rel[2,:] = ic3+ic4

    return u_rel

def U_relax_12(neq,nk,l,v,xc,u): # single positive jet -- smoothed

    u_rel = np.zeros((neq,nk))
 
    a1 = 0.1
    a2 = 0.1
    lj = 0.05*l
    ic3 = v*(1+np.tanh(4*(xc-(l/2-l/16))/lj + 2))*(1-np.tanh(4*(xc-(l/2+l/16))/lj - 2))/4

    u_rel[2,:] = ic3

    return u_rel
