
import numpy as np
from pyscf import lib
from pyscf.lib import logger

#einsum = np.einsum
einsum = lib.einsum


def cc_resp_Xi1(t1, t2, X_ov, X_oo, X_vv, X_vo):

    Xi1 = einsum('ea,ie->ia', X_vv, t1)
    Xi1 -= einsum('im,ma->ia', X_oo, t1)
    tmp1 = einsum('em,ie->im', X_vo, t1)
    Xi1 -= einsum('im,ma->ia', tmp1, t1)
    Xi1 += einsum('em,imae->ia', X_vo, t2)
    Xi1 += X_ov

    return Xi1


def cc_resp_Xi2(t1, t2, X_oo, X_vv, X_vo):

    tmp1 = einsum('fb,ijaf->ijab', X_vv, t2)
    tmp2 = einsum('jm,imab->ijab', X_oo, t2)
    tmp1 = tmp1 - tmp1.transpose(0,1,3,2)
    tmp2 = tmp2 - tmp2.transpose(1,0,2,3)

    tmp3 = einsum('em,ie->im', X_vo, t1)
    tmp4 = einsum('im,mjab->ijab', tmp3, t2)
    tmp4 = tmp4 - tmp4.transpose(1,0,2,3)

    tmp5 = einsum('fm,ma->fa', X_vo, t1)
    tmp6 = einsum('fa,ijfb->ijab', tmp5, t2)
    tmp6 = tmp6 - tmp6.transpose(0,1,3,2)

    Xi2 = tmp1 - tmp2 -tmp4 -tmp6
    
    return Xi2

def cc_resp_etax1(l1,l2,t1,t2,X_oo, X_vv, X_vo, X_ov):

    etax1 = einsum('ae,ie->ai', X_vv, l1)
    etax1 -= einsum('mi,ma->ai', X_oo, l1)
    tmp1 = einsum('me,ie->mi',t1,l1)
    etax1 -= einsum('am,mi->ai',X_vo,tmp1)
    tmp2 = einsum('ei,me->mi',X_vo,t1)
    etax1 -= einsum('mi,ma->ai',tmp2,l1)
    tmp3 = einsum('mnfe,mife->ni',t2,l2)
    etax1 -= 0.5 * einsum('ni,an->ai',tmp3,X_vo)
    tmp4 = einsum('nmfe,nmfa->ae',t2,l2)
    etax1 -= 0.5 * einsum('ae,ei->ai',tmp4,X_vo)

    etax1 += X_vo
    return etax1


def cc_resp_etax2(l1,l2,t1,t2,X_ov, X_oo, X_vv, X_vo):
    tmp1 = einsum('ia,bj->abij', l1, X_vo)
    tmp1 = tmp1 - tmp1.transpose(0,1,3,2)
    etax2 = tmp1 - tmp1.transpose(1,0,2,3)

    tmp2 = einsum('ijae,be->abij',l2, X_vv)
    etax2 += tmp2 - tmp2.transpose(1,0,2,3)
    tmp3 = einsum('imab,mj->abij',l2, X_oo)
    etax2 -= tmp3 - tmp3.transpose(0,1,3,2)

    tmp4 = einsum('me,ijae->amij',t1, l2)
    tmp5 = einsum('amij,bm->abij',tmp4, X_vo)
    etax2 -= tmp5 - tmp5.transpose(1,0,2,3)

    tmp6 = einsum('me,imab->abie',t1, l2)
    tmp7 = einsum('abie,ej->abij',tmp6, X_vo)
    etax2 -= tmp7 - tmp7.transpose(0,1,3,2)

    return etax2


def cc_resp_tF1(tx1, tx2, eris, t1, t2, l1, l2, 
                Foo, Fvv, Fov, 
                Woooo, Wvvvv, Wovvo, Wvovv, Wovoo, Wvvvo, Wooov):
    
#Term 1 V_ca,ki * tx_ia  
    tF1 = einsum('kica,ia->ck', eris.oovv, tx1)

#Term 2 -l_ci * F_ak * tx_ia
    tmp1 = einsum('ic,ka->caik', l1, Fov)
    tF1 -= einsum('caik,ia->ck', tmp1, tx1)

#Term 3 -l_ak * F_ci * tx_ia
    tmp2 = einsum('ka,ic->acki', l1, Fov)
    tF1 -= einsum('acki,ia->ck', tmp2, tx1)

#Term 4 l_ek * W_caei * tx_ia
    tmp3 = einsum('ke,eica->caki',l1, Wvovv)
    tF1 += einsum('caki,ia->ck', tmp3, tx1)

#Term 5  -l_cm * W_maki * tx_ia
    tmp4 = einsum('mc,kima->caki',l1, Wooov)
    tF1 -= einsum('caki,ia->ck', tmp4, tx1)

#Term 6 -l_am * W_mcik * tx_ia
    tmp5 = einsum('ma,ikmc->acik',l1, Wooov)
    tF1 -= einsum('acik,ia->ck', tmp5, tx1)

#Term 7 l_ei * W_acek * tx_ia
    tmp6 = einsum('ie,ekac->acik',l1, Wvovv)
    tF1 += einsum('acik,ia->ck', tmp6, tx1)

#Term 8 -1/2 * l_ci * V_abkj * tx2_ijab
    tmp7 = einsum('kjab,ijab->ik',eris.oovv, tx2)
    tF1 -= 0.5 * einsum('ic,ik->ck', l1, tmp7)

#Term 9 -1/2 * l_ak * V_cbij * tx2_ijab
    tmp8 = einsum('ijcb,ijab->ca',eris.oovv, tx2)
    tF1 -= 0.5 * einsum('ka,ca->ck', l1, tmp8)

#Term 10 l_ai * V_bcjk * tx2_ijab
    tmp9 = einsum('jkbc,ijab->icak',eris.oovv, tx2)
    tF1 += einsum('ia,icak->ck', l1, tmp9)

# #Term 11 -l_aemk * W_mcie * tx1_ia
    tmp10 = einsum('mkae,ia->mkie',l2, tx1)
    tF1 += einsum('mkie,iecm->ck',tmp10, Wovvo)

#Term 12 -l_ecim * W_amek * tx1_ia
    tmp11 = einsum('ia,keam->keim',tx1, Wovvo)
    tF1 += einsum('keim,imec->ck',tmp11, l2)

#Term 13 0.5 * l_efki * W_caef * tx1_ia
    tmp12 = einsum('kief,efca->caki',l2, Wvvvv)
    tF1 += 0.5 * einsum('caki,ia->ck',tmp12, tx1)

# #Term 14 0.5 * l_camn * W_mnki * tx1_ia
    tmp13 = einsum('mnca,kimn->caki',l2, Woooo)
    tF1 += 0.5 * einsum('caki,ia->ck',tmp13, tx1)

#Term 15 -0.5 * l_ecmn * t_mnef * V_faki * tx_ia
    tmp14 = einsum('mnec,mnef->cf', l2, t2)
    tmp15 = einsum('cf,kifa->caki', tmp14, eris.oovv)
    tF1 -= 0.5 * einsum('caki,ia->ck',tmp15, tx1)

#Term 16 -0.5 * l_efmk * t_mnef * V_cani * tx_ia
    tmp16 = einsum('mkef,mnef->nk', l2, t2)
    tmp17 = einsum('nk,nica->caki', tmp16, eris.oovv)
    tF1 -= 0.5 * einsum('caki,ia->ck',tmp17, tx1)

#Term 17 -0.5 * l_eamn * t_mnef * V_fcik * tx_ia
    tmp18 = einsum('mnea,mnef->af', l2, t2)
    tmp19 = einsum('af,ikfc->acik', tmp18, eris.oovv)
    tF1 -= 0.5 * einsum('acik,ia->ck',tmp19, tx1)

#Term 18 -0.5 * l_efmi * t_mnef * V_acnk * tx_ia
    tmp20 = einsum('mief,mnef->ni', l2, t2)
    tmp21 = einsum('ni,nkac->acik', tmp20, eris.oovv)
    tF1 -= 0.5 * einsum('acik,ia->ck',tmp21, tx1)

#Term 19 -0.5 * l_acij * F_bk * tx2_ijab
    tmp22 = einsum('kb,ijab->ijak', Fov, tx2)
    tF1 -= 0.5 * einsum('ijac,ijak->ck', l2, tmp22)

#Term 20 -0.5 * l_abik * F_cj * tx2_ijab
    tmp23 = einsum('jc,ijab->icab', Fov, tx2)
    tF1 -= 0.5 * einsum('ikab,icab->ck', l2, tmp23)

#Term 21 -l_ebik * W_acej * tx2_ijab
    tmp24 = einsum('ejac,ijab->iceb', Wvovv, tx2)
    tF1 -= einsum('ikeb,iceb->ck', l2, tmp24)

#Term 22 l_acmj * W_mbik * tx2_ijab
    tmp25 = einsum('ikmb,ijab->mjak', Wooov, tx2)
    tF1 += einsum('mjac,mjak->ck', l2, tmp25)


#Term 23 -0.25 * l_ecij * W_abek * tx2_ijab
    tmp26 = einsum('ekab,ijab->ijek', Wvovv, tx2)
    tF1 -= 0.25 * einsum('ijec,ijek->ck', l2, tmp26)

#Term 24 0.25 * l_abmk * W_mcij * tx2_ijab
    tmp27 = einsum('ijmc,ijab->mcab', Wooov, tx2)
    tF1 += 0.25 * einsum('mkab,mcab->ck', l2, tmp27)

#Term 25 0.5 * l_aeij * W_bcek * tx2_ijab
    tmp28 = einsum('ijae,ijab->eb', l2, tx2)
    tF1 += 0.5 * einsum('eb,ekbc->ck', tmp28, Wvovv)

#Term 26 -0.5 * l_abim * W_mcjk * tx2_ijab
    tmp29 = einsum('imab,ijab->jm', l2, tx2)
    tF1 -= 0.5 * einsum('jm,jkmc->ck', tmp29, Wooov)

    return tF1


def cc_resp_tF2(tx1, tx2, eris, t1, t2, l1, l2, 
                Foo, Fvv, Fov, 
                Woooo, Wvvvv, Wovvo, Wvovv, Wovoo, Wvvvo, Wooov):
    
#Term 1 P_cd P_kl l_ck * V_dali * tx1_ia
    tmp1 = einsum('lida,ia->dl', eris.oovv, tx1)
    tmp2 = einsum('kc,dl->cdkl', l1, tmp1)
    tmp2 = tmp2 -tmp2.transpose(1,0,2,3)
    tF2 = tmp2 -tmp2.transpose(0,1,3,2)

#Term 2 -Pkl l_ak * V_cdil * tx1_ia
    tmp3 = einsum('ilcd,ia->cdal', eris.oovv, tx1)
    tmp4 = einsum('ka,cdal->cdkl', l1, tmp3)
    tF2 -= tmp4 -tmp4.transpose(0,1,3,2)

#Term 3 -Pcd l_ci * V_adkl * tx1_ia
    tmp5 = einsum('klad,ia->idkl', eris.oovv, tx1)
    tmp6 = einsum('ic,idkl->cdkl', l1, tmp5)
    tF2 -= tmp6 -tmp6.transpose(1,0,2,3)

#Term 4 -Pkl l_cdki * F_al * tx1_ia
    tmp7 = einsum('la,ia->il', Fov, tx1)
    tmp8 = einsum('kicd,il->cdkl', l2, tmp7)
    tF2 -= tmp8 -tmp8.transpose(0,1,3,2)

#Term 5 -Pcd l_cakl * F_di * tx1_ia
    tmp9 = einsum('id,ia->da', Fov, tx1)
    tmp10 = einsum('klca,da->cdkl', l2, tmp9)
    tF2 -= tmp10 -tmp10.transpose(1,0,2,3)

#Term 6 Pcd l_cekl * W_daei * tx1_ia
    tmp11 = einsum('eida,ia->de', Wvovv, tx1)
    tmp12 = einsum('klce,de->cdkl', l2, tmp11)
    tF2 += tmp12 -tmp12.transpose(1,0,2,3)

#Term 7 -Pkl l_cdkm * W_mali * tx1_ia
    tmp13 = einsum('lima,ia->ml', Wooov, tx1)
    tmp14 = einsum('kmcd,ml->cdkl', l2, tmp13)
    tF2 -= tmp14 -tmp14.transpose(0,1,3,2)

#Term 8 Pcd Pkl l_caml * W_mdki * tx1_ia
    tmp15 = einsum('kimd,ia->mdka', Wooov, tx1)
    tmp16 = einsum('mlca,mdka->cdkl', l2, tmp15)
    tmp16 = tmp16 - tmp16.transpose(0,1,3,2)
    tF2 += tmp16 - tmp16.transpose(1,0,2,3)

#Term 9 l_cdmi * W_makl * tx1_ia
    tmp17 = einsum('klma,ia->mikl', Wooov, tx1)
    tF2 += einsum('micd,mikl->cdkl', l2, tmp17)

#Term 10 -Pcd Pkl l_edki * W_cael * tx1_ia
    tmp18 = einsum('elca,ia->ciel', Wvovv, tx1)
    tmp19 = einsum('kied,ciel->cdkl', l2, tmp18)
    tmp19 = tmp19 - tmp19.transpose(0,1,3,2)
    tF2 -= tmp19 - tmp19.transpose(1,0,2,3)

#Term 11 -l_eakl * W_cdei * tx1_ia
    tmp20 = einsum('eicd,ia->cdea', Wvovv, tx1)
    tF2 -= einsum('klea,cdea->cdkl', l2, tmp20)

#Term 12 -0.5 * Pcd l_cakl * V_dbij * tx2_ijab
    tmp21 = einsum('ijdb,ijab->da', eris.oovv, tx2)
    tmp22 = einsum('klca,da->cdkl', l2, tmp21)
    tF2 -= 0.5* (tmp22 - tmp22.transpose(1,0,2,3))

#Term 13 -0.5 * Pkl l_cdki * V_ablj * tx2_ijab
    tmp23 = einsum('ljab,ijab->il', eris.oovv, tx2)
    tmp24 = einsum('kicd,il->cdkl', l2, tmp23)
    tF2 -= 0.5* (tmp24 - tmp24.transpose(0,1,3,2))

#Term 14 0.25 * l_cdij * V_abkl * tx2_ijab
    tmp25 = einsum('klab,ijab->ijkl', eris.oovv, tx2)
    tmp26 = einsum('ijcd,ijkl->cdkl', l2, tmp25)
    tF2 += 0.25 * tmp26 

#Term 15 0.25 * l_abkl * V_cdij * tx2_ijab
    tmp27 = einsum('ijcd,ijab->cdab', eris.oovv, tx2)
    tmp28 = einsum('klab,cdab->cdkl', l2, tmp27)
    tF2 += 0.25 * tmp28

#Term 16 Pkl Pcd l_cbkj * V_dali * tx2_ijab
    tmp29 = einsum('lida,ijab->djlb',eris.oovv, tx2)
    tmp30 = einsum('kjcb,djlb->cdkl', l2, tmp29)
    tmp30 = tmp30 - tmp30.transpose(0,1,3,2)
    tF2 += tmp30 - tmp30.transpose(1,0,2,3)

#Term 17 -0.5 * Pkl l_abik * V_cdjl * tx2_ijab
    tmp31 = einsum('ikab,ijab->jk',l2, tx2)
    tmp32 = einsum('jk,jlcd->cdkl', tmp31, eris.oovv)
    tF2 -= 0.5* (tmp32 - tmp32.transpose(0,1,3,2))

#Term 18 -0.5 * Pcd l_acij * V_bdkl * tx2_ijab
    tmp33 = einsum('ijac,ijab->cb',l2, tx2)
    tmp34 = einsum('cb,klbd->cdkl', tmp33, eris.oovv)
    tF2 -= 0.5* (tmp34 - tmp34.transpose(1,0,2,3))

    return tF2
