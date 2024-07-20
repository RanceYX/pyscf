import numpy as np

from pyscf import lib
from pyscf.lib import logger, module_method
from pyscf.cc import ccsd
from pyscf.cc import eom_rccsd, eom_gccsd
from pyscf.cc import gccsd
from pyscf.cc import gintermediates as imd
from pyscf.cc import gintermediates_response as imd_rsp


def lrccsd(lr, t1, t2, l1, l2, oprA, oprB, mo_coeff, omega, eris=None, imds=None, **kwargs):

    if imds is None:
        imds = lr.make_imds(eris)                 #get EOM-EE intermediates

    nocc = lr.nocc
    nmo = lr.nmo
    eris = imds.eris

    oprA_mo = Prop_MO()
    oprB_mo = Prop_MO()
    oprA_mo.prop_ao2mo(nocc, oprA, mo_coeff)       # Transform propA int from ao2mo
    oprB_mo.prop_ao2mo(nocc, oprB, mo_coeff)       # Transform propB int from ao2mo

    imds_rspA = IMDS_rsp()
    imds_rspB = IMDS_rsp()
    imds_rspA.make_lr(oprA_mo, t1, t2, l1, l2)   #get LR-RSP intermediates
    imds_rspB.make_lr(oprB_mo, t1, t2, l1, l2)   #get LR-RSP intermediates

    tx_oprA_pw = Tx()
    tx_oprB_pw = Tx()
    tx_oprA_mw = Tx()
    tx_oprB_mw = Tx()

    tx_oprA_pw = lr.make_tx(imds, nocc, nmo, imds_rspA, omega)  #get txA(+w)
    tx_oprB_pw = lr.make_tx(imds, nocc, nmo, imds_rspB, omega)  #get txB(+w)
    tx_oprA_mw = lr.make_tx(imds, nocc, nmo, imds_rspA, -omega) #get txA(-w)
    tx_oprB_mw = lr.make_tx(imds, nocc, nmo, imds_rspB, -omega) #get txB(-w) 

    tx_oprA_pw.make_lrtxF(imds, t1, t2, l1, l2, eris)  #get txA(+w)
    tx_oprB_pw.make_lrtxF(imds, t1, t2, l1, l2, eris)  #get txB(+w)
    tx_oprA_mw.make_lrtxF(imds, t1, t2, l1, l2, eris) #get txA(-w)
    tx_oprB_mw.make_lrtxF(imds, t1, t2, l1, l2, eris) #get txB(-w) 


    lr_rsp = lr.get_lrrsp(tx_oprA_pw, tx_oprB_pw, tx_oprA_mw, tx_oprB_mw, 
                          imds, imds_rspA, imds_rspB) #Compute linear-response function
    
    return lr_rsp 



vector_to_amplitudes_ee = ccsd.vector_to_amplitudes_s4
amplitudes_to_vector_ee = ccsd.amplitudes_to_vector_s4

def eeccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Wang, Tu, and Wang, J. Chem. Theory Comput. 10, 5567 (2014) Eqs.(9)-(10)
    # Note: Last line in Eq. (10) is superfluous.
    # See, e.g. Gwaltney, Nooijen, and Barlett, Chem. Phys. Lett. 248, 189 (1996)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    # Eq. (9)
    Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += lib.einsum('me,imae->ia', imds.Fov, r2)
    Hr1 += lib.einsum('maei,me->ia', imds.Wovvo, r1)
    Hr1 -= 0.5*lib.einsum('mnie,mnae->ia', imds.Wooov, r2)
    Hr1 += 0.5*lib.einsum('amef,imef->ia', imds.Wvovv, r2)
    # Eq. (10)
    tmpab = lib.einsum('be,ijae->ijab', imds.Fvv, r2)
    tmpab -= 0.5*lib.einsum('mnef,ijae,mnbf->ijab', imds.Woovv, imds.t2, r2)
    tmpab -= lib.einsum('mbij,ma->ijab', imds.Wovoo, r1)
    tmpab -= lib.einsum('amef,ijfb,me->ijab', imds.Wvovv, imds.t2, r1)
    tmpij  = lib.einsum('mj,imab->ijab', -imds.Foo, r2)
    tmpij -= 0.5*lib.einsum('mnef,imab,jnef->ijab', imds.Woovv, imds.t2, r2)
    tmpij += lib.einsum('abej,ie->ijab', imds.Wvvvo, r1)
    tmpij += lib.einsum('mnie,njab,me->ijab', imds.Wooov, imds.t2, r1)

    tmpabij = lib.einsum('mbej,imae->ijab', imds.Wovvo, r2)
    tmpabij = tmpabij - tmpabij.transpose(1,0,2,3)
    tmpabij = tmpabij - tmpabij.transpose(0,1,3,2)
    Hr2 = tmpabij

    Hr2 += tmpab - tmpab.transpose(0,1,3,2)
    Hr2 += tmpij - tmpij.transpose(1,0,2,3)
    Hr2 += 0.5*lib.einsum('mnij,mnab->ijab', imds.Woooo, r2)
    Hr2 += 0.5*lib.einsum('abef,ijef->ijab', imds.Wvvvv, r2)

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector


def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = np.zeros((nocc,nvir), dtype=t1.dtype)
    Hr2 = np.zeros((nocc,nocc,nvir,nvir), dtype=t1.dtype)
 
    for i in range(nocc):
        for a in range(nvir):
            Hr1[i,a] = imds.Fvv[a,a] - imds.Foo[i,i] 
               
    for a in range(nvir):
        for b in range(nvir):
            for i in range(nocc):
                for j in range(nocc):
                    Hr2[j,i,b,a] = (
                        imds.Fvv[a,a]+
                        imds.Fvv[b,b]-
                        imds.Foo[j,j]-
                        imds.Foo[i,i]
                    )

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def lrccsd_Xi_trial(eom, imds_rsp=None):
    Xi1 = imds_rsp.Xi1
    nocc, nvir = Xi1.shape

    Hr1 = np.zeros((nocc,nvir), dtype=Xi1.dtype)
    Hr2 = np.zeros((nocc,nocc,nvir,nvir), dtype=Xi1.dtype)
    for i in range(nocc):
        for a in range(nvir):
            Hr1[i,a] = Xi1[i,a]
    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector


def property_gradient(lr, vector, imds_rsp=None):
    nocc = lr.nocc
    nmo = lr.nmo
    r1, r2 = lr.vector_to_amplitudes(vector, nmo, nocc)

    Xir1 = -1.0 * lib.einsum('ia,ia->',  r1, imds_rsp.Xi1)
    Xir2 = -0.25 * lib.einsum('ijab,ijab->',  r2, imds_rsp.Xi2)
    vector = np.array([Xir1 + Xir2])
    return vector


def get_XiVec(lr, imds_rsp=None):
    vector = lr.amplitudes_to_vector(imds_rsp.Xi1, imds_rsp.Xi2)
    return vector


class LRCCCC(eom_gccsd.EOMEE):

    kernel = lrccsd
    lrccsd = lrccsd
    matvec = eeccsd_matvec
    get_diag = eeccsd_diag
    propgrd = property_gradient
    get_Xi_trial = lrccsd_Xi_trial
    XiVec = get_XiVec


    amplitudes_to_vector = staticmethod(amplitudes_to_vector_ee)
    vector_to_amplitudes = module_method(vector_to_amplitudes_ee,
                                         absences=['nmo', 'nocc'])

    def get_trial(self, nroots, diag=None):
        if diag is None:
            diag = self.Xi_trial()
 
        idx = diag.argsort()

        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)

        return guess


    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds
    

    def make_tx(self, imds, nocc, nmo, imds_rsp, omega):  
        tx_opr = Tx()
        matvec, propgrd, diag = self.gen_matvec(imds, imds_rsp)   #get sigma vector
        Xi_trial = self.get_Xi_trial(imds_rsp)
        nroots = 1
        guess = self.get_trial(nroots, Xi_trial)
        XiVecs = self.XiVec(imds_rsp)

        solution = lib.davidson_nosym1_linear_system    #davidson procedure
        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)
        
        print ('Start Davidson procedure, omega=', omega)
        tx_opr_vecs = solution(matvec, guess, propgrd, XiVecs, omega, precond,
                           tol=1e-12, max_cycle=50,
                           max_space=100, nroots=1, verbose=None)
        
        tx1, tx2 = self.vector_to_amplitudes(tx_opr_vecs, nmo, nocc)
        tx_opr.tx1 = tx1
        tx_opr.tx2 = tx2

        print ('Davidson finished')
        print('t1x norm: ', np.linalg.norm(tx1)**2)
        print('t2x norm: ', (np.linalg.norm(tx2)/2)**2)

        return tx_opr

    def gen_matvec(self, imds=None, imds_rsp=None):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        propgrd = lambda xs: [self.propgrd(x, imds_rsp) for x in xs]
        return matvec, propgrd, diag
        

    def get_lrrsp(self, tx_oprA_pw, tx_oprB_pw, tx_oprA_mw, tx_oprB_mw, 
                    imds, imds_rspA, imds_rspB):

        eta_A1 =  imds_rspA.etax1
        eta_A2 =  imds_rspA.etax2
        eta_B1 =  imds_rspB.etax1
        eta_B2 =  imds_rspB.etax2


        lrrsp_pw = lib.einsum('ai,ia->', eta_A1, tx_oprB_pw.tx1)
        lrrsp_pw += 0.25 * lib.einsum('abij,ijab->', eta_A2, tx_oprB_pw.tx2)
        lrrsp_pw += 0.5 * lib.einsum('ai,ia->',  tx_oprA_mw.txF1, tx_oprB_pw.tx1)
        lrrsp_pw += 0.125 * lib.einsum('abij,ijab->',  tx_oprA_mw.txF2, tx_oprB_pw.tx2)

        lrrsp_pw += lib.einsum('ai,ia->', eta_B1, tx_oprA_pw.tx1)
        lrrsp_pw += 0.25 * lib.einsum('abij,ijab->', eta_B2, tx_oprA_pw.tx2)
        lrrsp_pw += 0.5 * lib.einsum('ai,ia->',  tx_oprB_mw.txF1, tx_oprA_pw.tx1)
        lrrsp_pw += 0.125 * lib.einsum('abij,ijab->',  tx_oprB_mw.txF2, tx_oprA_pw.tx2)

        lrrsp_mw = lib.einsum('ai,ia->', eta_A1, tx_oprB_mw.tx1)
        lrrsp_mw += 0.25 * lib.einsum('abij,ijab->', eta_A2, tx_oprB_mw.tx2)
        lrrsp_mw += 0.5 * lib.einsum('ai,ia->',  tx_oprA_pw.txF1, tx_oprB_mw.tx1)
        lrrsp_mw += 0.125 * lib.einsum('abij,ijab->',  tx_oprA_pw.txF2, tx_oprB_mw.tx2)

        lrrsp_mw += lib.einsum('ai,ia->', eta_B1, tx_oprA_mw.tx1)
        lrrsp_mw += 0.25 * lib.einsum('abij,ijab->', eta_B2, tx_oprA_mw.tx2)
        lrrsp_mw += 0.5 * lib.einsum('ai,ia->',  tx_oprB_pw.txF1, tx_oprA_mw.tx1)
        lrrsp_mw += 0.125 * lib.einsum('abij,ijab->',  tx_oprB_pw.txF2, tx_oprA_mw.tx2)

        lrrsp = 0.5* (lrrsp_pw + np.conjugate(lrrsp_mw))
        return lrrsp


class Prop_MO:

    def __init__(self):
        self.oo = None
        self.ov = None
        self.vo = None
        self.vv = None


    def prop_ao2mo(self, nocc, opr_ao, mo_coeff):
        nao = mo_coeff.shape[1] // 2
        opr_ao_re = np.zeros((2*nao, 2*nao),dtype=np.complex128)
        for n in range(nao):
            opr_ao_re[n][:nao] = opr_ao[n]
        for n in range(nao, 2*nao):
            opr_ao_re[n][nao:] = opr_ao[n-nao]

        opr_mo_all = np.einsum('ip,ij,jq->pq', mo_coeff, opr_ao_re, mo_coeff)
        self.oo = opr_mo_all[:nocc, :nocc]
        self.ov = opr_mo_all[:nocc, nocc:]
        self.vo = opr_mo_all[nocc:, :nocc]
        self.vv = opr_mo_all[nocc:, nocc:]

        return self


class _IMDS:

    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared = False
        self.made_ee_imds = False

    def make_ee(self):
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris

        self.Foo_bar = imd.cc_Foo(t1, t2, eris)
        self.Fvv_bar = imd.cc_Fvv(t1, t2, eris)
        self.Fov_bar = imd.cc_Fov(t1, t2, eris)

        self.Foo = imd.Foo(t1, t2, eris)
        self.Fvv = imd.Fvv(t1, t2, eris)
        self.Fov = imd.Fov(t1, t2, eris)

        print('Foo norm: ', np.linalg.norm(self.Foo)**2)
        print('Fvv norm: ', np.linalg.norm(self.Fvv)**2)
        print('Fov norm: ', np.linalg.norm(self.Fov)**2)
        print('Foo_bar norm: ', np.linalg.norm(self.Foo_bar)**2)
        print('Fvv_bar norm: ', np.linalg.norm(self.Fvv_bar)**2)
        print('Fov_bar norm: ', np.linalg.norm(self.Fov_bar)**2)

        # 2 virtuals
        self.Wovvo = imd.Wovvo(t1, t2, eris)
        self.Woovv = eris.oovv
        
        self.Woooo = imd.Woooo(t1, t2, eris)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)

        self.Wvovv = imd.Wvovv(t1, t2, eris)
        self.Wvvvv = imd.Wvvvv(t1, t2, eris)
        self.Wvvvo = imd.Wvvvo(t1, t2, eris,self.Wvvvv)

        print('Wovvo norm: ', (np.linalg.norm(self.Wovvo))**2)
        print('Woovv norm: ', (np.linalg.norm(self.Woovv))**2)
        print('Woooo norm: ', (np.linalg.norm(self.Woooo))**2)
        print('Wooov norm: ', (np.linalg.norm(self.Wooov))**2)
        print('Wovoo norm: ', (np.linalg.norm(self.Wovoo))**2)
        print('Wvovv norm: ', (np.linalg.norm(self.Wvovv))**2)
        print('Wvvvv norm: ', (np.linalg.norm(self.Wvvvv))**2)
        print('Wvvvo norm: ', (np.linalg.norm(self.Wvvvo))**2)

        self.made_ee_imds = True
        logger.timer(self, 'EOM-CCSD EE intermediates', *cput0)
        return self
    

class IMDS_rsp:
    def __init__(self):
        self.Xi1 = None
        self.Xi2 = None
        self.etax1 = None
        self.etax2 = None


    def make_lr(self, opr, t1, t2, l1, l2):

        X_oo = opr.oo
        X_ov = opr.ov
        X_vo = opr.vo
        X_vv = opr.vv
        
        self.Xi1 = imd_rsp.cc_resp_Xi1(t1,t2, X_ov, X_oo, X_vv, X_vo)
        self.Xi2 = imd_rsp.cc_resp_Xi2(t1,t2, X_oo, X_vv, X_vo)
        self.etax1 = imd_rsp.cc_resp_etax1(l1,l2,t1,t2, X_oo, X_vv, X_vo, X_ov)
        self.etax2 = imd_rsp.cc_resp_etax2(l1,l2,t1,t2, X_ov, X_oo, X_vv, X_vo)

        return self
    

class Tx(LRCCCC):
    def __init__(self):
        self.tx1 = None
        self.tx2 = None
        self.txF1 = None
        self.txF2 = None


    def make_lrtxF(self, imd_ee, t1, t2, l1, l2, eris):

        tx1, tx2 = self.tx1, self.tx2

        Foo = imd_ee.Foo_bar
        Fov = imd_ee.Fov_bar
        Fvv = imd_ee.Fvv_bar

        Wovvo = imd_ee.Wovvo
        Woooo = imd_ee.Woooo
        Wovoo = imd_ee.Wovoo
        Wvovv = imd_ee.Wvovv
        Wvvvv = imd_ee.Wvvvv
        Wvvvo = imd_ee.Wvvvo
        Wooov = imd_ee.Wooov

        self.txF1 = imd_rsp.cc_resp_tF1(tx1, tx2, eris, t1, t2, l1, l2, 
                Foo, Fvv, Fov, Woooo, Wvvvv, Wovvo, Wvovv, Wovoo, Wvvvo, Wooov)
        self.txF2 = imd_rsp.cc_resp_tF2(tx1, tx2, eris, t1, t2, l1, l2, 
                Foo, Fvv, Fov, Woooo, Wvvvv, Wovvo, Wvovv, Wovoo, Wvvvo, Wooov)
        
        return self
    

