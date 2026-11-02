import numpy as np
import pandas as pd

def GetMinimisation(effPromptList, effFDList, rawYieldList, effPromptUncList, effFDUncList,
                    rawYieldUncList, corr=True, precision=1.e-8, nMaxIter=100):
    '''
    Method to retrieve prompt and FD corrected yields with an analytic system minimisation

    Parameters
    ----------
    - effPromptList: list of efficiencies for prompt D
    - effFDList: list of efficiencies for FD D
    - rawYieldList: list of raw yields
    - effPromptUncList: list of uncertainties on efficiencies for prompt D
    - effFDUncList: list of uncertainties on efficiencies for FD D
    - rawYieldUncList: list of uncertainties on raw yields
    - corr (bool, optional): whether to compute the correlation
    - precision (float, optional): target precision for minimisation procedure
    - nMaxIter (int, optional): max number of iterations for minimisation procedure

    Returns
    ----------
    - mCorrYield (numpy matrix): corrected yields (Nprompt, NFD)
    - mCovariance (numpy matrix): covariance matrix for corrected yields
    - redChiSquare (float): reduced chi square
    - dicOfMatrices (dictionary): dictionary with all matrices used in minimisation procedure
    '''
    nCutSets = len(effPromptList)

    mRawYield = np.zeros(shape=(nCutSets, 1))
    mEff = np.zeros(shape=(nCutSets, 2))
    mCovSets = np.zeros(shape=(nCutSets, nCutSets))
    mCorrSets = np.zeros(shape=(nCutSets, nCutSets))
    mWeights = np.zeros(shape=(nCutSets, nCutSets))

    mCorrYield = np.zeros(shape=(2, 1))
    mCorrYieldOld = np.zeros(shape=(2, 1))
    mCovariance = np.zeros(shape=(2, 2))
    mRes = np.zeros(shape=(nCutSets, 1))

    for iCutSet, (rawYield, effPrompt, effFD) in enumerate(zip(rawYieldList, effPromptList, effFDList)):
        # mRawYield.itemset(iCutSet, rawYield)
        # mEff.itemset((iCutSet, 0), effPrompt)
        # mEff.itemset((iCutSet, 1), effFD)
        # NumPy > 2.0
        mRawYield[iCutSet] = rawYield
        mEff[iCutSet, 0] = effPrompt
        mEff[iCutSet, 1] = effFD

    mRawYield = np.matrix(mRawYield)
    mEff = np.matrix(mEff)

    for iIter in range(nMaxIter):
        if iIter == 0:
            # mCorrYield.itemset(0, 0)
            # mCorrYield.itemset(1, 0)
            # NumPy > 2.0
            mCorrYield[0] = 0
            mCorrYield[1] = 0
        for iCutSetRow, (rawYieldUncRow, effPromptUncRow, effFDUncRow) in enumerate(\
            zip(rawYieldUncList, effPromptUncList, effFDUncList)):
            for iCutSetCol, (rawYieldUncCol, effPromptUncCol, effFDUncCol) in enumerate(\
                zip(rawYieldUncList, effPromptUncList, effFDUncList)):
                # uncRow = np.sqrt(rawYieldUncRow**2 + effPromptUncRow**2 *
                #                  mCorrYield.item(0)**2 + effFDUncRow**2 * mCorrYield.item(1)**2)
                # NumPy > 2.0
                uncRow = np.sqrt(rawYieldUncRow**2 + effPromptUncRow**2 *
                                 mCorrYield[0]**2 + effFDUncRow**2 * mCorrYield[1]**2)
                
                # uncCol = np.sqrt(rawYieldUncCol**2 + effPromptUncCol**2 *
                #                  mCorrYield.item(0)**2 + effFDUncCol**2 * mCorrYield.item(1)**2)
                # NumPy > 2.0
                uncCol = np.sqrt(rawYieldUncCol**2 + effPromptUncCol**2 *
                                 mCorrYield[0]**2 + effFDUncCol**2 * mCorrYield[1]**2)
                if corr and uncRow > 0 and uncCol > 0:
                    if uncRow < uncCol:
                        rho = uncRow / uncCol
                    else:
                        rho = uncCol / uncRow
                else:
                    if iCutSetRow == iCutSetCol:
                        rho = 1.
                    else:
                        rho = 0.
                covRowCol = rho * uncRow * uncCol
                # mCovSets.itemset((iCutSetRow, iCutSetCol), covRowCol)
                # mCorrSets.itemset((iCutSetRow, iCutSetCol), rho)
                # NumPy > 2.0
                mCovSets[iCutSetRow, iCutSetCol] = covRowCol
                mCorrSets[iCutSetRow, iCutSetCol] = rho

        mCovSets = np.matrix(mCovSets)
        mWeights = np.linalg.inv(np.linalg.cholesky(mCovSets))
        mWeights = mWeights.T * mWeights
        mEffT = mEff.T

        mCovariance = (mEffT * mWeights) * mEff
        mCovariance = np.linalg.inv(np.linalg.cholesky(mCovariance))
        mCovariance = mCovariance.T * mCovariance

        mCorrYield = mCovariance * (mEffT * mWeights) * mRawYield
        mRes = mEff * mCorrYield - mRawYield
        mResT = np.transpose(mRes)

        # if (mCorrYield.item(0)-mCorrYieldOld.item(0)) / mCorrYield.item(0) < precision and \
        #     (mCorrYield.item(1)-mCorrYieldOld.item(1)) / mCorrYield.item(1) < precision:
        #     break
        # NumPy > 2.0
        if (mCorrYield[0]-mCorrYieldOld[0]) / mCorrYield[0] < precision and \
           (mCorrYield[1]-mCorrYieldOld[1]) / mCorrYield[1] < precision:
            break

        mCorrYieldOld = np.copy(mCorrYield)

    #reduced chi2
    redChiSquare = mResT * mWeights * mRes / (nCutSets - 2)
    #dictionary with matrices used in minimisation procedure
    dicOfMatrices = {'covMatrix':mCovSets, 'weightMatrix':mWeights, 'corrMatrix':mCorrSets}

    return mCorrYield, mCovariance, float(redChiSquare), dicOfMatrices

def GetPromptFDFractionCutSet(accEffPrompt, accEffFD, corrYieldPrompt, corrYieldFD,
                              covPromptPrompt, covFDFD, covPromptFD):
    '''
    Helper method to get the prompt and FD fractions for a given cut set with the cut-variation method
    The Uncertainties on the efficiencies are neglected

    Parameters
    ----------
    - accEffPrompt: acc x eff for prompt
    - accEffFD: acc x eff for prompt
    - corrYieldPrompt: corr yield for prompt from cut-variation method
    - corrYieldFD: corr yield for FD from cut-variation method
    - covPromptPrompt: covariance for corrected yields (prompt, prompt) component
    - covFDFD: covariance for corrected yields (FD, FD) component
    - covPromptFD: covariance for corrected yields (prompt, FD) component

    Returns
    ----------
    - frac: list of two elements with prompt and FD fractions
    - uncFrac: list of two elements with uncertainties on prompt and FD fractions
    '''

    # prompt fraction
    fPrompt = accEffPrompt * corrYieldPrompt / (accEffPrompt * corrYieldPrompt + accEffFD * corrYieldFD)
    defPdeNP = (accEffPrompt * (accEffPrompt * corrYieldPrompt + accEffFD * corrYieldFD) - accEffPrompt**2
                * corrYieldPrompt) / (accEffPrompt * corrYieldPrompt + accEffFD * corrYieldFD)**2
    defPdeNF = - accEffFD * accEffPrompt * corrYieldPrompt / \
        (accEffPrompt * corrYieldPrompt + accEffFD * corrYieldFD)**2
    fPromptUnc = np.sqrt(defPdeNP**2 * covPromptPrompt + defPdeNF**2 * covFDFD + 2 * defPdeNP * defPdeNF * covPromptFD)

    # feed-down fraction
    fFD = accEffFD * corrYieldFD / (accEffPrompt * corrYieldPrompt + accEffFD * corrYieldFD)
    defFdeNF = (accEffFD * (accEffFD * corrYieldFD + accEffPrompt * corrYieldPrompt) - accEffFD**2
                * corrYieldFD) / (accEffPrompt * corrYieldPrompt + accEffFD * corrYieldFD)**2
    defFdeNP = - accEffFD * accEffPrompt * corrYieldFD / \
        (accEffPrompt * corrYieldPrompt + accEffFD * corrYieldFD)**2
    fFDUnc = np.sqrt(defFdeNF**2 * covFDFD + defFdeNP**2 * covPromptPrompt + 2 * defFdeNF * defFdeNP * covPromptFD)

    frac = [fPrompt, fFD]
    uncFrac = [fPromptUnc, fFDUnc]

    return frac, uncFrac