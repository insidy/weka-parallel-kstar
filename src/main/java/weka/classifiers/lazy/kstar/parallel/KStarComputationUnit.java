package weka.classifiers.lazy.kstar.parallel;

import weka.classifiers.lazy.kstar.KStarCache;
import weka.classifiers.lazy.kstar.KStarConstants;
import weka.classifiers.lazy.kstar.KStarNominalAttribute;
import weka.classifiers.lazy.kstar.KStarNumericAttribute;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.concurrent.Callable;

public class KStarComputationUnit implements Callable<KStarComputationResults>, KStarConstants {
    Instances m_Data;
    Instance m_ToCompare;
    int m_ClassType;
    int m_NumClasses;
    int m_NumAttributes;

    KStarConfig m_Configuration;

    protected KStarCache[] m_Cache;

    public KStarComputationUnit(Instance toCompare, Instances computationData, KStarConfig configuration, KStarCache[] cache) {
        m_Data = computationData;
        m_ToCompare = toCompare;
        m_Configuration = configuration;
        m_Cache = cache;

        m_ClassType = m_Data.classAttribute().type();
        m_NumClasses = m_Data.numClasses();
        m_NumAttributes = m_Data.numAttributes();
    }
    @Override
    public KStarComputationResults call() throws Exception {
        KStarComputationResults result = new KStarComputationResults(m_NumClasses);

        double transProb = 0.0;
        Instance trainInstance;
        Enumeration enu = m_Data.enumerateInstances();
        while (enu.hasMoreElements()) {
            trainInstance = (Instance) enu.nextElement();
            transProb = instanceTransformationProbability(m_ToCompare, trainInstance);
            switch (m_ClassType) {
                case Attribute.NOMINAL:
                    result.classProbability[(int) trainInstance.classValue()] += transProb;
                    break;
                case Attribute.NUMERIC:
                    result.predictedValue[0] += transProb * trainInstance.classValue();
                    result.temp += transProb;
                    break;
            }
        }

        return result;
    }

    protected double instanceTransformationProbability(Instance first,
                                                       Instance second) {
        String debug = "(KStar.instanceTransformationProbability) ";
        double transProb = 1.0;
        int numMissAttr = 0;
        for (int i = 0; i < m_NumAttributes; i++) {
            if (i == m_Data.classIndex()) {
                continue; // ignore class attribute
            }
            if (first.isMissing(i)) { // test instance attribute value is missing
                numMissAttr++;
                continue;
            }
            transProb *= attrTransProb(first, second, i);
            // normilize for missing values
            if (numMissAttr != m_NumAttributes) {
                transProb = Math.pow(transProb, (double) m_NumAttributes /
                        (m_NumAttributes - numMissAttr));
            } else { // weird case!
                transProb = 0.0;
            }
        }
        // normilize for the train dataset
        return transProb / m_Configuration.m_TotalInstances;
    }

    /**
     * Calculates the transformation probability of the indexed test attribute
     * to the indexed train attribute.
     *
     * @param first  the test instance.
     * @param second the train instance.
     * @param col    the index of the attribute in the instance.
     * @return the value of the transformation probability.
     */
    private double attrTransProb(Instance first, Instance second, int col) {
        String debug = "(KStar.attrTransProb)";
        double transProb = 0.0;
        KStarNominalAttribute ksNominalAttr;
        KStarNumericAttribute ksNumericAttr;
        switch (m_Data.attribute(col).type()) {
            case Attribute.NOMINAL:
                ksNominalAttr = new KStarNominalAttribute(first, second, col, m_Data,
                        m_Configuration.m_RandClassCols,
                        m_Cache[col]);
                ksNominalAttr.setOptions(m_Configuration.m_MissingMode, m_Configuration.m_BlendMethod, m_Configuration.m_GlobalBlend);
                transProb = ksNominalAttr.transProb();
                ksNominalAttr = null;
                break;

            case Attribute.NUMERIC:
                ksNumericAttr = new KStarNumericAttribute(first, second, col,
                        m_Data, m_Configuration.m_RandClassCols,
                        m_Cache[col]);
                ksNumericAttr.setOptions(m_Configuration.m_MissingMode, m_Configuration.m_BlendMethod, m_Configuration.m_GlobalBlend);
                transProb = ksNumericAttr.transProb();
                ksNumericAttr = null;
                break;
        }
        return transProb;
    }

    public static class KStarConfig {
        public int[][] m_RandClassCols;
        public int m_TotalInstances;
        public int m_MissingMode = M_AVERAGE;
        public int m_BlendMethod = B_SPHERE;
        public int m_GlobalBlend = 20;

    }
}
