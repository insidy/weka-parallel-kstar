/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    KStar.java
 *    Copyright (C) 1995-97 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.lazy;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.lazy.kstar.KStarCache;
import weka.classifiers.lazy.kstar.KStarConstants;
import weka.classifiers.lazy.kstar.KStarNominalAttribute;
import weka.classifiers.lazy.kstar.KStarNumericAttribute;
import weka.classifiers.lazy.kstar.gpu.KStarOpenCLProxy;
import weka.classifiers.lazy.kstar.parallel.KStarComputationResults;
import weka.classifiers.lazy.kstar.parallel.KStarComputationUnit;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.*;
import java.util.concurrent.*;

/**
 * <!-- globalinfo-start -->
 * K* is an instance-based classifier, that is the class of a test instance is based upon the class of those training instances similar to it, as determined by some similarity function.  It differs from other instance-based learners in that it uses an entropy-based distance function.<br/>
 * <br/>
 * For more information on K*, see<br/>
 * <br/>
 * John G. Cleary, Leonard E. Trigg: K*: An Instance-based Learner Using an Entropic Distance Measure. In: 12th International Conference on Machine Learning, 108-114, 1995.
 * <p/>
 * <!-- globalinfo-end -->
 * <p/>
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Cleary1995,
 *    author = {John G. Cleary and Leonard E. Trigg},
 *    booktitle = {12th International Conference on Machine Learning},
 *    pages = {108-114},
 *    title = {K*: An Instance-based Learner Using an Entropic Distance Measure},
 *    year = {1995}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * <p/>
 * <!-- options-start -->
 * Valid options are: <p/>
 * <p/>
 * <pre> -B &lt;num&gt;
 *  Manual blend setting (default 20%)
 * </pre>
 * <p/>
 * <pre> -E
 *  Enable entropic auto-blend setting (symbolic class only)
 * </pre>
 * <p/>
 * <pre> -M &lt;char&gt;
 *  Specify the missing value treatment mode (default a)
 *  Valid options are: a(verage), d(elete), m(axdiff), n(ormal)
 * </pre>
 * <p/>
 * <!-- options-end -->
 *
 * @author Len Trigg (len@reeltwo.com)
 * @author Abdelaziz Mahoui (am14@cs.waikato.ac.nz) - Java port
 * @version $Revision: 5525 $
 */
public class ParallelKStar
        extends AbstractClassifier
        implements KStarConstants, UpdateableClassifier, TechnicalInformationHandler {

    /**
     * for serialization
     */
    static final long serialVersionUID = 332458330800479083L;

    /**
     * The training instances used for classification.
     */
    protected Instances m_Train;

    /**
     * The number of instances in the dataset
     */
    protected int m_NumInstances;

    /**
     * The number of class values
     */
    protected int m_NumClasses;

    /**
     * The number of attributes
     */
    protected int m_NumAttributes;

    /**
     * The class attribute type
     */
    protected int m_ClassType;

    /**
     * Table of random class value colomns
     */
    protected int[][] m_RandClassCols;

    /**
     * Flag turning on and off the computation of random class colomns
     */
    protected int m_ComputeRandomCols = ON;

    /**
     * Flag turning on and off the initialisation of config variables
     */
    protected int m_InitFlag = ON;

    /**
     * A custom data structure for caching distinct attribute values
     * and their scale factor or stop parameter.
     */
    protected KStarCache[] m_Cache;

    /**
     * missing value treatment
     */
    protected int m_MissingMode = M_AVERAGE;

    /**
     * 0 = use specified blend, 1 = entropic blend setting
     */
    protected int m_BlendMethod = B_SPHERE;

    /**
     * default sphere of influence blend setting
     */
    protected int m_GlobalBlend = 20;

    /** Number of simultaneous threads to use in computation (0 = autodetect | CPU + 1). */
    protected int m_NumThreads = 0;
    private static final int CPU_COUNT = Runtime.getRuntime().availableProcessors();
    private static final int CORE_POOL_SIZE = CPU_COUNT + 1;

    /* Open CL mode*/
    public static final int NO_OPENCL = 0;
    public static final int OPENCL_GPU = 1;
    public static final int OPENCL_CPU = 2;

    public static final Tag[] TAGS_OPENCL = {
            new Tag(NO_OPENCL, "Do not use OpenCL"),
            new Tag(OPENCL_GPU, "OpenCL GPU mode"),
            new Tag(OPENCL_CPU, "OpenCL CPU mode")
    };

    protected int m_OpenCLMode = NO_OPENCL;

    /**
     * Define possible missing value handling methods
     */
    public static final Tag[] TAGS_MISSING = {
            new Tag(M_DELETE, "Ignore the instances with missing values"),
            new Tag(M_MAXDIFF, "Treat missing values as maximally different"),
            new Tag(M_NORMAL, "Normalize over the attributes"),
            new Tag(M_AVERAGE, "Average column entropy curves")
    };


    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {

        return "K* is an instance-based classifier, that is the class of a test "
                + "instance is based upon the class of those training instances "
                + "similar to it, as determined by some similarity function.  It differs "
                + "from other instance-based learners in that it uses an entropy-based "
                + "distance function.\n\n"
                + "For more information on K*, see\n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "John G. Cleary and Leonard E. Trigg");
        result.setValue(Field.TITLE, "K*: An Instance-based Learner Using an Entropic Distance Measure");
        result.setValue(Field.BOOKTITLE, "12th International Conference on Machine Learning");
        result.setValue(Field.YEAR, "1995");
        result.setValue(Field.PAGES, "108-114");

        return result;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.DATE_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data
     * @throws Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances instances) throws Exception {
        String debug = "(KStar.buildClassifier) ";

        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        m_Train = new Instances(instances, 0, instances.numInstances());

        // initializes class attributes ** java-speaking! :-) **
        init_m_Attributes();
    }

    /**
     * Adds the supplied instance to the training set
     *
     * @param instance the instance to add
     * @throws Exception if instance could not be incorporated successfully
     */
    public void updateClassifier(Instance instance) throws Exception {
        String debug = "(KStar.updateClassifier) ";

        if (m_Train.equalHeaders(instance.dataset()) == false)
            throw new Exception("Incompatible instance types");
        if (instance.classIsMissing())
            return;
        m_Train.add(instance);
        // update relevant attributes ...
        update_m_Attributes();
    }

    public double[] distributionForInstanceInOpenCL(Instance instance) throws Exception {
        String debug = "(KStar.distributionForInstance) ";
        double transProb = 0.0, temp = 0.0;
        double[] classProbability = new double[m_NumClasses];
        double[] predictedValue = new double[1];

        // initialization ...
        for (int i = 0; i < classProbability.length; i++) {
            classProbability[i] = 0.0;
        }
        predictedValue[0] = 0.0;
        if (m_InitFlag == ON) {
            // need to compute them only once and will be used for all instances.
            // We are doing this because the evaluation module controls the calls.
            if (m_BlendMethod == B_ENTROPY) {
                generateRandomClassColomns();
            }
            m_Cache = new KStarCache[m_NumAttributes];
            for (int i = 0; i < m_NumAttributes; i++) {
                m_Cache[i] = new KStarCache();
            }
            m_InitFlag = OFF;
            //      System.out.println("Computing...");
        }
        // init done.



            KStarComputationUnit.KStarConfig threadConfig = new KStarComputationUnit.KStarConfig();
            threadConfig.m_TotalInstances = m_NumInstances;
            threadConfig.m_BlendMethod = m_BlendMethod;
            threadConfig.m_GlobalBlend = m_GlobalBlend;
            threadConfig.m_MissingMode = m_MissingMode;
            threadConfig.m_RandClassCols = m_RandClassCols;
            KStarOpenCLProxy openClProxy = new KStarOpenCLProxy(instance, m_Train, m_NumClasses, m_NumAttributes, m_Train.numInstances(), threadConfig, m_OpenCLMode);



            KStarComputationResults results = openClProxy.compute();
            classProbability = results.classProbability;
            predictedValue = results.predictedValue;
            temp = results.temp;


        if (m_ClassType == Attribute.NOMINAL) {
            double sum = Utils.sum(classProbability);
            if (sum <= 0.0)
                for (int i = 0; i < classProbability.length; i++)
                    classProbability[i] = (double) 1 / (double) m_NumClasses;
            else Utils.normalize(classProbability, sum);
            return classProbability;
        } else {
            predictedValue[0] = (temp != 0) ? predictedValue[0] / temp : 0.0;
            return predictedValue;
        }
    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @throws Exception if an error occurred during the prediction
     */
    // parallel variables start


    // parallel variables end

    public double[] distributionForInstance(Instance instance) throws Exception {
        if( m_OpenCLMode != NO_OPENCL ) { // OpenCL for now..
            return distributionForInstanceInOpenCL(instance);
        }

        String debug = "(KStar.distributionForInstance) ";
        double transProb = 0.0, temp = 0.0;
        double[] classProbability = new double[m_NumClasses];
        double[] predictedValue = new double[1];

        // initialization ...
        for (int i = 0; i < classProbability.length; i++) {
            classProbability[i] = 0.0;
        }
        predictedValue[0] = 0.0;
        if (m_InitFlag == ON) {
            // need to compute them only once and will be used for all instances.
            // We are doing this because the evaluation module controls the calls.
            if (m_BlendMethod == B_ENTROPY) {
                generateRandomClassColomns();
            }
            m_Cache = new KStarCache[m_NumAttributes];
            for (int i = 0; i < m_NumAttributes; i++) {
                m_Cache[i] = new KStarCache();
            }
            m_InitFlag = OFF;
            //      System.out.println("Computing...");
        }
        // init done.
        Instance trainInstance;
        Enumeration enu = m_Train.enumerateInstances();


        if( m_Debug ) { // non parallel
            KStarComputationResults results = computeDirectly(instance);
            classProbability = results.classProbability;
            predictedValue = results.predictedValue;
        } else {
            int threads = getNumThreads() == 0 ?  CORE_POOL_SIZE : getNumThreads();
            ExecutorService sExecutor = Executors.newFixedThreadPool(threads);

            List<Instances> threadSplitInstances = getInstancesForEachThread(threads);
            List<Callable<KStarComputationResults>> computationUnits = new ArrayList<Callable<KStarComputationResults>>();

            KStarComputationUnit.KStarConfig threadConfig = new KStarComputationUnit.KStarConfig();
            threadConfig.m_TotalInstances = m_NumInstances;
            threadConfig.m_BlendMethod = m_BlendMethod;
            threadConfig.m_GlobalBlend = m_GlobalBlend;
            threadConfig.m_MissingMode = m_MissingMode;
            threadConfig.m_RandClassCols = m_RandClassCols;
            for(Instances computationData : threadSplitInstances) { // scatter
                computationUnits.add(new KStarComputationUnit(instance, computationData, threadConfig, m_Cache.clone()));
            }



            List<Future<KStarComputationResults>> results = sExecutor.invokeAll(computationUnits);
            sExecutor.shutdown();
            sExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS); // "gather wait"
            for(Future<KStarComputationResults> result : results) {
                KStarComputationResults threadComputationResult = result.get();
                predictedValue[0] += threadComputationResult.predictedValue[0];
                temp += threadComputationResult.temp;

                for (int i = 0; i < classProbability.length; i++) {
                    classProbability[i] += threadComputationResult.classProbability[i];
                }
            }

        }


        if (m_ClassType == Attribute.NOMINAL) {
            double sum = Utils.sum(classProbability);
            if (sum <= 0.0)
                for (int i = 0; i < classProbability.length; i++)
                    classProbability[i] = (double) 1 / (double) m_NumClasses;
            else Utils.normalize(classProbability, sum);
            return classProbability;
        } else {
            predictedValue[0] = (temp != 0) ? predictedValue[0] / temp : 0.0;
            return predictedValue;
        }
    }

    //// Parallel start

    private KStarComputationResults computeDirectly(Instance instance) {
        KStarComputationResults result = new KStarComputationResults(m_NumClasses);

        double transProb = 0.0, temp = 0.0;
        Instance trainInstance;
        Enumeration enu = m_Train.enumerateInstances();
        while (enu.hasMoreElements()) {
            trainInstance = (Instance) enu.nextElement();
            transProb = instanceTransformationProbability(instance, trainInstance);
            switch (m_ClassType) {
                case Attribute.NOMINAL:
                    result.classProbability[(int) trainInstance.classValue()] += transProb;
                    break;
                case Attribute.NUMERIC:
                    result.predictedValue[0] += transProb * trainInstance.classValue();
                    temp += transProb;
                    break;
            }
        }

        return result;
    }


    private List<Instances> getInstancesForEachThread(int threads) {
        ArrayList<Instances> threadInstances = new ArrayList<Instances>();
        int splitInstancesIn = threads;

        int instancesForEachThread = m_Train.numInstances() / splitInstancesIn;
        for(int i = 0; i < splitInstancesIn; i++) {
            int offset = i * instancesForEachThread;
            int numberOfInstances = instancesForEachThread;

            if(i == (splitInstancesIn - 1) ) { // last thread
                numberOfInstances = numberOfInstances + (m_Train.numInstances() - (offset + numberOfInstances)); // add any remaining
            }

            threadInstances.add(new Instances(m_Train, offset, numberOfInstances));
        }

        return threadInstances;
    }

    //// Parallel end

    /**
     * Calculate the probability of the first instance transforming into the
     * second instance:
     * the probability is the product of the transformation probabilities of
     * the attributes normilized over the number of instances used.
     *
     * @param first  the test instance
     * @param second the train instance
     * @return transformation probability value
     */
    private double instanceTransformationProbability(Instance first,
                                                     Instance second) {
        String debug = "(KStar.instanceTransformationProbability) ";
        double transProb = 1.0;
        int numMissAttr = 0;
        for (int i = 0; i < m_NumAttributes; i++) {
            if (i == m_Train.classIndex()) {
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
        return transProb / m_NumInstances;
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
        switch (m_Train.attribute(col).type()) {
            case Attribute.NOMINAL:
                ksNominalAttr = new KStarNominalAttribute(first, second, col, m_Train,
                        m_RandClassCols,
                        m_Cache[col]);
                ksNominalAttr.setOptions(m_MissingMode, m_BlendMethod, m_GlobalBlend);
                transProb = ksNominalAttr.transProb();
                ksNominalAttr = null;
                break;

            case Attribute.NUMERIC:
                ksNumericAttr = new KStarNumericAttribute(first, second, col,
                        m_Train, m_RandClassCols,
                        m_Cache[col]);
                ksNumericAttr.setOptions(m_MissingMode, m_BlendMethod, m_GlobalBlend);
                transProb = ksNumericAttr.transProb();
                ksNumericAttr = null;
                break;
        }
        return transProb;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String missingModeTipText() {
        return "Determines how missing attribute values are treated.";
    }

    /**
     * Gets the method to use for handling missing values. Will be one of
     * M_NORMAL, M_AVERAGE, M_MAXDIFF or M_DELETE.
     *
     * @return the method used for handling missing values.
     */
    public SelectedTag getMissingMode() {

        return new SelectedTag(m_MissingMode, TAGS_MISSING);
    }

    /**
     * Sets the method to use for handling missing values. Values other than
     * M_NORMAL, M_AVERAGE, M_MAXDIFF and M_DELETE will be ignored.
     *
     * @param newMode the method to use for handling missing values.
     */
    public void setMissingMode(SelectedTag newMode) {

        if (newMode.getTags() == TAGS_MISSING) {
            m_MissingMode = newMode.getSelectedTag().getID();
        }
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector optVector = new Vector(3);
        optVector.addElement(new Option(
                "\tManual blend setting (default 20%)\n",
                "B", 1, "-B <num>"));
        optVector.addElement(new Option(
                "\tEnable entropic auto-blend setting (symbolic class only)\n",
                "E", 0, "-E"));
        optVector.addElement(new Option(
                "\tSpecify the missing value treatment mode (default a)\n"
                        + "\tValid options are: a(verage), d(elete), m(axdiff), n(ormal)\n",
                "M", 1, "-M <char>"));
        optVector.addElement(new Option(
                "\tThe number of simultaneous threads to use for computation, 0 for autodetect.\n"
                        + "\t(default 0)",
                "threads", 1, "-threads <num>"));
        optVector.addElement(new Option(
                "\tUse of OpenCL Parallel computation mode. 0 = No OpenCL (defualt), 1 = OpenCL GPU, 2 = OpenCL CPU\n"
                        + "\t(default 0)",
                "opencl", 1, "-opencl <char>"));

        return optVector.elements();
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalBlendTipText() {
        return "The parameter for global blending. Values are restricted to [0,100].";
    }

    /**
     * Set the global blend parameter
     *
     * @param b the value for global blending
     */
    public void setGlobalBlend(int b) {
        m_GlobalBlend = b;
        if (m_GlobalBlend > 100) {
            m_GlobalBlend = 100;
        }
        if (m_GlobalBlend < 0) {
            m_GlobalBlend = 0;
        }
    }

    /**
     * Get the value of the global blend parameter
     *
     * @return the value of the global blend parameter
     */
    public int getGlobalBlend() {
        return m_GlobalBlend;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String entropicAutoBlendTipText() {
        return "Whether entropy-based blending is to be used.";
    }

    /**
     * Set whether entropic blending is to be used.
     *
     * @param e true if entropic blending is to be used
     */
    public void setEntropicAutoBlend(boolean e) {
        if (e) {
            m_BlendMethod = B_ENTROPY;
        } else {
            m_BlendMethod = B_SPHERE;
        }
    }

    /**
     * Get whether entropic blending being used
     *
     * @return true if entropic blending is used
     */
    public boolean getEntropicAutoBlend() {
        if (m_BlendMethod == B_ENTROPY) {
            return true;
        }

        return false;
    }

    /**
     * Gets the OpenCL mode for Parallel computation.
     *
     * @return the method used for handling missing values.
     */
    public SelectedTag getOpenCLMode() {

        return new SelectedTag(m_OpenCLMode, TAGS_OPENCL);
    }

    /**
     * Sets if Parallel KStar should use OpenCL.
     *
     * @param newMode OpenCL computation mode.
     */
    public void setOpenCLMode(SelectedTag newMode) {

        if (newMode.getTags() == TAGS_OPENCL) {
            m_OpenCLMode = newMode.getSelectedTag().getID();
        }
    }

    /**
     * Get the number of simultaneous threads used in training, 0 for autodetect.
     *
     * @return the maximum depth.
     */
    public int getNumThreads(){
        return m_NumThreads;
    }

    /**
     * Set the number of simultaneous threads used in training, 0 for autodetect.
     *
     * @param value the maximum depth.
     */
    public void setNumThreads(int value){
        m_NumThreads = value;
    }
    /**
     * Parses a given list of options. <p/>
     * <p/>
     * <!-- options-start -->
     * Valid options are: <p/>
     * <p/>
     * <pre> -B &lt;num&gt;
     *  Manual blend setting (default 20%)
     * </pre>
     * <p/>
     * <pre> -E
     *  Enable entropic auto-blend setting (symbolic class only)
     * </pre>
     * <p/>
     * <pre> -M &lt;char&gt;
     *  Specify the missing value treatment mode (default a)
     *  Valid options are: a(verage), d(elete), m(axdiff), n(ormal)
     * </pre>
     * <p/>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        String debug = "(KStar.setOptions)";
        String blendStr = Utils.getOption('B', options);
        if (blendStr.length() != 0) {
            setGlobalBlend(Integer.parseInt(blendStr));
        }

        setEntropicAutoBlend(Utils.getFlag('E', options));

        String threadStr = Utils.getOption("threads", options);
        if (threadStr.length() != 0) {
            setNumThreads(Integer.parseInt(threadStr));
        }

        String openClStr = Utils.getOption("opencl", options);
        if (openClStr.length() != 0) {
            switch (openClStr.charAt(0)) {
                case '0':
                    setOpenCLMode(new SelectedTag(NO_OPENCL, TAGS_OPENCL));
                    break;
                case '1':
                    setOpenCLMode(new SelectedTag(OPENCL_GPU, TAGS_OPENCL));
                    break;
                case '2':
                    setOpenCLMode(new SelectedTag(OPENCL_CPU, TAGS_OPENCL));
                    break;
                default:
                    setOpenCLMode(new SelectedTag(NO_OPENCL, TAGS_OPENCL));
            }
        }

        String missingModeStr = Utils.getOption('M', options);
        if (missingModeStr.length() != 0) {
            switch (missingModeStr.charAt(0)) {
                case 'a':
                    setMissingMode(new SelectedTag(M_AVERAGE, TAGS_MISSING));
                    break;
                case 'd':
                    setMissingMode(new SelectedTag(M_DELETE, TAGS_MISSING));
                    break;
                case 'm':
                    setMissingMode(new SelectedTag(M_MAXDIFF, TAGS_MISSING));
                    break;
                case 'n':
                    setMissingMode(new SelectedTag(M_NORMAL, TAGS_MISSING));
                    break;
                default:
                    setMissingMode(new SelectedTag(M_AVERAGE, TAGS_MISSING));
            }
        }
        Utils.checkForRemainingOptions(options);
    }


    /**
     * Gets the current settings of K*.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions() {
        // -B <num> -E -M <char>
        String[] options = new String[7];
        int itr = 0;
        options[itr++] = "-B";
        options[itr++] = "" + m_GlobalBlend;

        if (getEntropicAutoBlend()) {
            options[itr++] = "-E";
        }

        options[itr++] = "-M";
        if (m_MissingMode == M_AVERAGE) {
            options[itr++] = "" + "a";
        } else if (m_MissingMode == M_DELETE) {
            options[itr++] = "" + "d";
        } else if (m_MissingMode == M_MAXDIFF) {
            options[itr++] = "" + "m";
        } else if (m_MissingMode == M_NORMAL) {
            options[itr++] = "" + "n";
        }

        if(getNumThreads() > 0){
            options[itr++] = "-threads";
            options[itr++] = String.valueOf(getNumThreads());
        }

        while (itr < options.length) {
            options[itr++] = "";
        }
        return options;
    }

    /**
     * Returns a description of this classifier.
     *
     * @return a description of this classifier as a string.
     */
    public String toString() {
        StringBuffer st = new StringBuffer();
        st.append("Parallel KStar Beta Version (0.1b).\n"
                + "Copyright (c) 1995-97 by Len Trigg (trigg@cs.waikato.ac.nz).\n"
                + "Java port to Weka by Abdelaziz Mahoui (am14@cs.waikato.ac.nz).\n"
                + "Parallel version by Paulo César Büttenbender. \n\nParallel KStar options : ");
        String[] ops = getOptions();
        for (int i = 0; i < ops.length; i++) {
            st.append(ops[i] + ' ');
        }
        return st.toString();
    }

    /**
     * Main method for testing this class.
     *
     * @param argv should contain command line options (see setOptions)
     */
    public static void main(String[] argv) {
        runClassifier(new ParallelKStar(), argv);
    }

    /**
     * Initializes the m_Attributes of the class.
     */
    private void init_m_Attributes() {
        try {
            m_NumInstances = m_Train.numInstances();
            m_NumClasses = m_Train.numClasses();
            m_NumAttributes = m_Train.numAttributes();
            m_ClassType = m_Train.classAttribute().type();
            m_InitFlag = ON;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Updates the m_attributes of the class.
     */
    private void update_m_Attributes() {
        m_NumInstances = m_Train.numInstances();
        m_InitFlag = ON;
    }

    /**
     * Note: for Nominal Class Only!
     * Generates a set of random versions of the class colomn.
     */
    private void generateRandomClassColomns() {
        String debug = "(KStar.generateRandomClassColomns)";
        Random generator = new Random(42);
        //    Random generator = new Random();
        m_RandClassCols = new int[NUM_RAND_COLS + 1][];
        int[] classvals = classValues();
        for (int i = 0; i < NUM_RAND_COLS; i++) {
            // generate a randomized version of the class colomn
            m_RandClassCols[i] = randomize(classvals, generator);
        }
        // original colomn is preserved in colomn NUM_RAND_COLS
        m_RandClassCols[NUM_RAND_COLS] = classvals;
    }

    /**
     * Note: for Nominal Class Only!
     * Returns an array of the class values
     *
     * @return an array of class values
     */
    private int[] classValues() {
        String debug = "(KStar.classValues)";
        int[] classval = new int[m_NumInstances];
        for (int i = 0; i < m_NumInstances; i++) {
            try {
                classval[i] = (int) m_Train.instance(i).classValue();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return classval;
    }

    /**
     * Returns a copy of the array with its elements randomly redistributed.
     *
     * @param array     the array to randomize.
     * @param generator the random number generator to use
     * @return a copy of the array with its elements randomly redistributed.
     */
    private int[] randomize(int[] array, Random generator) {
        String debug = "(KStar.randomize)";
        int index;
        int temp;
        int[] newArray = new int[array.length];
        System.arraycopy(array, 0, newArray, 0, array.length);
        for (int j = newArray.length - 1; j > 0; j--) {
            index = (int) (generator.nextDouble() * (double) j);
            temp = newArray[j];
            newArray[j] = newArray[index];
            newArray[index] = temp;
        }
        return newArray;
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 5525 $");
    }

} // class end

