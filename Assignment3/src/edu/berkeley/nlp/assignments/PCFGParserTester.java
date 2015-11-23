package edu.berkeley.nlp.assignments;

import edu.berkeley.nlp.io.PennTreebankReader;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.*;

import java.security.InvalidAlgorithmParameterException;
import java.util.*;

/**
 * Harness for PCFG Parser project.
 *
 * @author Dan Klein
 */
public class PCFGParserTester {

  /**
   * Parsers are required to map sentences to trees.  How a parser is constructed and trained is not specified.
   */
  static interface Parser {
    Tree<String> getBestParse(List<String> sentence);
  }

  /**
   * Baseline parser (though not a baseline I've ever seen before).  Tags the sentence using the baseline tagging
   * method, then either retrieves a known parse of that tag sequence, or builds a right-branching parse for unknown tag
   * sequences.
   */
  static class BaselineParser implements Parser {
    CounterMap<List<String>, Tree<String>> knownParses;
    CounterMap<Integer, String> spanToCategories;
    Lexicon lexicon;

    public Tree<String> getBestParse(List<String> sentence) {
      List<String> tags = getBaselineTagging(sentence);
      Tree<String> annotatedBestParse = null;
      if (knownParses.keySet().contains(tags)) {
        annotatedBestParse = getBestKnownParse(tags);
      } else {
        annotatedBestParse = buildRightBranchParse(sentence, tags);
      }
      return TreeAnnotations.unAnnotateTree(annotatedBestParse);
    }

    private Tree<String> buildRightBranchParse(List<String> words, List<String> tags) {
      int currentPosition = words.size() - 1;
      Tree<String> rightBranchTree = buildTagTree(words, tags, currentPosition);
      while (currentPosition > 0) {
        currentPosition--;
        rightBranchTree = merge(buildTagTree(words, tags, currentPosition), rightBranchTree);
      }
      rightBranchTree = addRoot(rightBranchTree);
      return rightBranchTree;
    }

    private Tree<String> merge(Tree<String> leftTree, Tree<String> rightTree) {
      int span = leftTree.getYield().size() + rightTree.getYield().size();
      String mostFrequentLabel = spanToCategories.getCounter(span).argMax();
      List<Tree<String>> children = new ArrayList<Tree<String>>();
      children.add(leftTree);
      children.add(rightTree);
      return new Tree<String>(mostFrequentLabel, children);
    }

    private Tree<String> addRoot(Tree<String> tree) {
      return new Tree<String>("ROOT", Collections.singletonList(tree));
    }

    private Tree<String> buildTagTree(List<String> words, List<String> tags, int currentPosition) {
      Tree<String> leafTree = new Tree<String>(words.get(currentPosition));
      Tree<String> tagTree = new Tree<String>(tags.get(currentPosition), Collections.singletonList(leafTree));
      return tagTree;
    }

    private Tree<String> getBestKnownParse(List<String> tags) {
      return knownParses.getCounter(tags).argMax();
    }

    private List<String> getBaselineTagging(List<String> sentence) {
      List<String> tags = new ArrayList<String>();
      for (String word : sentence) {
        String tag = getBestTag(word);
        tags.add(tag);
      }
      return tags;
    }

    private String getBestTag(String word) {
      double bestScore = Double.NEGATIVE_INFINITY;
      String bestTag = null;
      for (String tag : lexicon.getAllTags()) {
        double score = lexicon.scoreTagging(word, tag);
        if (bestTag == null || score > bestScore) {
          bestScore = score;
          bestTag = tag;
        }
      }
      return bestTag;
    }

    public BaselineParser(List<Tree<String>> trainTrees) {

      System.out.print("Annotating / binarizing training trees ... ");
      List<Tree<String>> annotatedTrainTrees = annotateTrees(trainTrees);
      System.out.println("done.");

      System.out.print("Building grammar ... ");
      Grammar grammar = new Grammar(annotatedTrainTrees);
      System.out.println("done. (" + grammar.getStates().size() + " states)");

      System.out.print("Discarding grammar and setting up a baseline parser ... ");
      lexicon = new Lexicon(annotatedTrainTrees);
      knownParses = new CounterMap<List<String>, Tree<String>>();
      spanToCategories = new CounterMap<Integer, String>();
      for (Tree<String> trainTree : annotatedTrainTrees) {
        List<String> tags = trainTree.getPreTerminalYield();
        knownParses.incrementCount(tags, trainTree, 1.0);
        tallySpans(trainTree, 0);
      }
      System.out.println("done.");
    }

    private List<Tree<String>> annotateTrees(List<Tree<String>> trees) {
      List<Tree<String>> annotatedTrees = new ArrayList<Tree<String>>();
      for (Tree<String> tree : trees) {
        annotatedTrees.add(TreeAnnotations.annotateTree(tree));
      }
      return annotatedTrees;
    }

    private int tallySpans(Tree<String> tree, int start) {
      if (tree.isLeaf() || tree.isPreTerminal()) return 1;
      int end = start;
      for (Tree<String> child : tree.getChildren()) {
        int childSpan = tallySpans(child, end);
        end += childSpan;
      }
      String category = tree.getLabel();
      if (!category.equals("ROOT"))
        spanToCategories.incrementCount(end - start, category, 1.0);
      return end - start;
    }
  }

  static class PreterminalTag{
    public boolean IsUnary = false;
    public String Tag;
    public Double Probability;
    public String Word;
    public PreterminalTag(String tag, double probability, String word){
      Tag = tag;
      Probability = probability;
      Word = word;
    }
  }

  static class Position {
    public Integer Row;
    public Integer Column;
    public Position(Integer row, Integer column){
      Row = row;
      Column = column;
    }
  }

  static class TagWithPosition{
    public String Tag;
    public Position TagPosition;
    public TagWithPosition(String tag, Position position){
      Tag = tag;
      TagPosition = position;
    }
  }

  static class Binary{
    public String ResultingTag;
    public TagWithPosition TagLeft;
    public TagWithPosition TagRight;
    public Double Probability;
    public boolean IsUnary = false;
    public Binary(String resultingTag, TagWithPosition tagLeft, TagWithPosition tagRight, double probability){
      ResultingTag = resultingTag;
      TagLeft = tagLeft;
      TagRight = tagRight;
      Probability = probability;
    }
  }

  static abstract class Cell{
    Integer Row;
    Integer Column;
    public Cell(Integer row, Integer column){
      BackPointers = new HashMap<String, String>();
      Column = column;
      Row = row;
    }
    public Map<String, String> BackPointers;

    public abstract Set<String> GetTags();
    public abstract double GetScore(String tag);

    public abstract void AddUnaryClosure(String parent, double probability);
  }

  static class PreterminalCell extends Cell{
    public HashMap<String, PreterminalTag> Preterminals;
    public PreterminalCell(HashMap<String, PreterminalTag> preterminals, Integer row, Integer column){
      super(row, column);
      Preterminals = preterminals;
    }

    @Override
    public Set<String> GetTags() {
      return Preterminals.keySet();
    }

    @Override
    public double GetScore(String tag) {
      PreterminalTag preterminalTag = Preterminals.getOrDefault(tag, null);
      if (preterminalTag != null){
        return preterminalTag.Probability;
      }
      return 0;
    }

    @Override
    public void AddUnaryClosure(String parent, double probability) {
      // Add a fake preterminal
      PreterminalTag fakePreterminal = new PreterminalTag(parent, probability, null);
      fakePreterminal.IsUnary = true;
      Preterminals.put(parent, fakePreterminal);
    }
  }

  static class BinaryCell extends Cell{
    public Map<String, Binary> Binaries;
    public BinaryCell(Map<String, Binary> binaries, Integer row, Integer column){
      super(row, column);
      Binaries = binaries;
    }

    @Override
    public Set<String> GetTags() {
      return Binaries.keySet();
    }

    @Override
    public double GetScore(String tag) {
      Binary binary = Binaries.getOrDefault(tag, null);
      if (binary != null){
        return binary.Probability;
      }
      return 0;
    }

    @Override
    public void AddUnaryClosure(String parent, double probability) {
      // Add a fake binary
      Binary fakeBinary = new Binary(parent, null, null, probability);
      fakeBinary.IsUnary = true;
      Binaries.put(parent, fakeBinary);
    }
  }

  /**
   * Baseline parser (though not a baseline I've ever seen before).  Tags the sentence using the baseline tagging
   * method, then either retrieves a known parse of that tag sequence, or builds a right-branching parse for unknown tag
   * sequences.
   */
  static class CKYParser implements Parser {
    Lexicon lexicon;
    Grammar grammar;
    UnaryClosure unaryClosures;

    public Tree<String> getBestParse(List<String> sentence) {
      System.out.println("GetBestParse for " + sentence.size() + " word sentence");

      // First row of the tree (bottom up)
      // Build the preterminal tags
      List<PreterminalCell> preterminalCells = getPreterminalTags(sentence);
      // Do unary rules to find extra possible tags
      AddUnaryRulesToPreterminalCells(preterminalCells);

      Cell[][] bottomUpTree = new Cell[sentence.size()][sentence.size()];
      int counterForFillingPreterminalCells = 0;
      for(PreterminalCell preterminalCell : preterminalCells){
        bottomUpTree[counterForFillingPreterminalCells][counterForFillingPreterminalCells] = preterminalCell;
        counterForFillingPreterminalCells++;
      }

      // Do binary linking
      int cellLevelCount = sentence.size();
      for(int cellLevelIndex = 1; cellLevelIndex < cellLevelCount; cellLevelIndex++){
        // For a specific level, traverse all the cells
        int cellCountForLevel = sentence.size() - cellLevelIndex;
        for(int cellIndex = 0; cellIndex < cellCountForLevel; cellIndex++){
          // On each cell we should find all possible combinations
          int cellRow = cellIndex;
          int cellColumn = cellLevelIndex + cellIndex;

          HashMap<String, Binary> binariesForCell = new HashMap<String, Binary>();

          for(int split = 0; split < cellLevelIndex; split++){
            int leftCellRow = cellRow;
            int leftCellColumn = cellColumn + split - cellLevelIndex;
            Cell leftCell = bottomUpTree[leftCellRow][leftCellColumn];
            int rightCellRow = cellRow + split + 1;
            int rightCellColumn = cellColumn;
            Cell rightCell = bottomUpTree[rightCellRow][rightCellColumn];

            AddBinariesForCellCombination(leftCell, rightCell, binariesForCell);

            // System.out.println("Left {" + leftCellRow + ", " + leftCellColumn + "} Right {" + rightCellRow + ", " + rightCellColumn + "}");
          }

          // At this point, binariesForCell contains all the possible binaries. We need to unary closure them, then write the cell
          BinaryCell binaryCell = new BinaryCell(binariesForCell, cellRow, cellColumn);
          AddUnaryRulesForCell(binaryCell);
          bottomUpTree[cellRow][cellColumn] = binaryCell;
        }
      }

      // Start the tree at root
      Tree<String> annotatedBestParse = new Tree<String>("ROOT", null);
      List<Tree<String>> rootChildren = new ArrayList<Tree<String>>();
      annotatedBestParse.setChildren(rootChildren);

      // Pick the beginning tag
      BinaryCell lastCellAsBinaryCell = as(BinaryCell.class, bottomUpTree[0][sentence.size() - 1]);
      PreterminalCell lastCellAsPreterminal = as(PreterminalCell.class, bottomUpTree[0][sentence.size() - 1]);
      if (lastCellAsBinaryCell != null){
        // Start with a binary
        Binary binary = GetMainRootTagFromBinaryCell(lastCellAsBinaryCell);

        // Build the beginning of the tree (always an S followed by something else
        Tree<String> sTree = new Tree<String>(binary.ResultingTag);
        rootChildren.add(sTree);

        // If we're here, it means we'll be able to build a tree. Just watch out for unaries and for the final preterminal transition
        AddBinaryToTree(lastCellAsBinaryCell, binary, sTree, bottomUpTree);
      } else {
        // This has to be a one word sentence
        PreterminalTag preterminal = GetMainRootTagFromPreterminalCell(lastCellAsPreterminal);
        Tree<String> sTree = new Tree<String>(preterminal.Tag);
        rootChildren.add(sTree);
        AddPreterminalToTree(lastCellAsPreterminal, sTree, bottomUpTree, preterminal);
      }
      return TreeAnnotations.unAnnotateTree(annotatedBestParse);
    }

    private PreterminalTag GetMainRootTagFromPreterminalCell(PreterminalCell lastCellAsPreterminal) {
      double maxProbability = 0;
      String maxTag = null;
      for(String tag : lastCellAsPreterminal.GetTags()){
        double probability = lastCellAsPreterminal.GetScore(tag);
        if (probability > maxProbability){
          maxTag = tag;
          maxProbability = probability;
        }
      }

      if (maxTag == null){
        // No tag is a weird case!
        throw new EmptyStackException();
      }

      return lastCellAsPreterminal.Preterminals.get(maxTag);
    }

    private Binary GetMainRootTagFromBinaryCell(BinaryCell lastCell) {
      // Normally you could start with S, but we're going to pick the most likely one instead
      if (lastCell.GetScore("S") > 0){
        return lastCell.Binaries.get("S");
      }

      // No S is a weird case!
      throw new EmptyStackException();


      /*double maxProbability = 0;
      String maxTag = null;
      for(String tag : lastCell.GetTags()){
        double currentProbability = lastCell.GetScore(tag);
        if (currentProbability > maxProbability){
          maxTag = tag;
          maxProbability = currentProbability;
        }
      }

      if (maxTag == null){
        // No probability is a bad case!
        throw new EmptyStackException();
      }

      return lastCell.Binaries.get(maxTag);*/
    }

    private void AddBinaryToTree(BinaryCell cell, Binary binary, Tree<String> currentTree, Cell[][] bottomUpTree) {
      List<Tree<String>> currentChildren = new ArrayList<Tree<String>>();
      currentTree.setChildren(currentChildren);

      while (binary.IsUnary){
        // Add as a unary - loop until we hit a non-unary
        Tree<String> unaryTree = new Tree<String>(binary.ResultingTag);
        currentChildren.add(unaryTree);
        currentChildren = new ArrayList<Tree<String>>();
        unaryTree.setChildren(currentChildren);
        String realTag = cell.BackPointers.get(binary.ResultingTag);
        binary = cell.Binaries.get(realTag);
      }

      // Now that it's not a unary, we can create a tree for the binary
      Tree<String> binaryTreeLeft = new Tree<String>(binary.TagLeft.Tag);
      Tree<String> binaryTreeRight = new Tree<String>(binary.TagRight.Tag);
      currentChildren.add(binaryTreeLeft);
      currentChildren.add(binaryTreeRight);

      // Look for the next cells and call concurrently
      Cell leftCell = bottomUpTree[binary.TagLeft.TagPosition.Row][binary.TagLeft.TagPosition.Column];
      Cell rightCell = bottomUpTree[binary.TagRight.TagPosition.Row][binary.TagRight.TagPosition.Column];

      AddTagFromCellInTree(leftCell, binaryTreeLeft, bottomUpTree, binary.TagLeft.Tag);
      AddTagFromCellInTree(rightCell, binaryTreeRight, bottomUpTree, binary.TagRight.Tag);
    }

    public static <T> T as(Class<T> t, Object o) {
      return t.isInstance(o) ? t.cast(o) : null;
    }

    private void AddTagFromCellInTree(Cell cell, Tree<String> tree, Cell[][] bottomUpTree, String tag) {
      BinaryCell binaryCell = as(BinaryCell.class, cell);
      PreterminalCell preterminalCell = as(PreterminalCell.class, cell);

      if (binaryCell != null){
        Binary binary = binaryCell.Binaries.get(tag);
        AddBinaryToTree(binaryCell, binary, tree, bottomUpTree);
      }
      else if (preterminalCell != null){
        PreterminalTag preterminalTag = preterminalCell.Preterminals.get(tag);
        AddPreterminalToTree(preterminalCell, tree, bottomUpTree, preterminalTag);
      }
    }

    private void AddPreterminalToTree(PreterminalCell cell, Tree<String> tree, Cell[][] bottomUpTree, PreterminalTag preterminalTag) {
      List<Tree<String>> currentChildren = new ArrayList<Tree<String>>();
      tree.setChildren(currentChildren);

      while (preterminalTag.IsUnary){
        // Add as a unary - loop until we hit a non-unary
        String realTag = cell.BackPointers.get(preterminalTag.Tag);
        preterminalTag = cell.Preterminals.get(realTag);

        Tree<String> unaryTree = new Tree<String>(preterminalTag.Tag);
        currentChildren.add(unaryTree);
        currentChildren = new ArrayList<Tree<String>>();
        unaryTree.setChildren(currentChildren);
      }

      // Now that it's not a unary, we can add the word
      Tree<String> terminalTree = new Tree<String>(preterminalTag.Word);
      currentChildren.add(terminalTree);
    }

    private void AddBinariesForCellCombination(Cell leftCell, Cell rightCell, HashMap<String, Binary> binariesForCell) {
      // Get all combinations of possible grammars between left and right
      Set<String> leftTags = leftCell.GetTags();

      for(String leftTag : leftTags){
        List<BinaryRule> binaryRules = grammar.getBinaryRulesByLeftChild(leftTag);
        for(BinaryRule binaryRule : binaryRules){
          double rightScore = rightCell.GetScore(binaryRule.getRightChild());
          if (rightScore > 0){
            double leftScore = leftCell.GetScore(leftTag);
            double probability = binaryRule.getScore() * leftScore * rightScore;
            // Add the binary combination
            // Check if the combination existed. If not, then add it. If yes, add it if it has a better score
            Binary existingBinary = binariesForCell.getOrDefault(binaryRule.getParent(), null);
            if (existingBinary == null || existingBinary.Probability < probability){
              binariesForCell.put(
                      binaryRule.getParent(),
                      new Binary(binaryRule.getParent(),
                              new TagWithPosition(binaryRule.getLeftChild(), new Position(leftCell.Row, leftCell.Column)),
                              new TagWithPosition(binaryRule.getRightChild(), new Position(rightCell.Row, rightCell.Column)),
                              probability));
            }
          }
        }
      }
    }

    private void AddUnaryRulesToPreterminalCells(List<PreterminalCell> preterminalCells) {
      System.out.println("Building unaries for first level");

      for(Cell cell : preterminalCells){
        AddUnaryRulesForCell(cell);
      }
    }

    private void AddUnaryRulesForCell(Cell cell) {
      boolean added = true;
      while(added){
        added = false;
        Set<String> set = new TreeSet<String>(cell.GetTags());
        for(String tagKey : set){
          List<UnaryRule> localClosures = unaryClosures.getClosedUnaryRulesByChild(tagKey);
          for(UnaryRule localClosure : localClosures){
            double tagProbability = cell.GetScore(tagKey);
            double probability = localClosure.getScore() * tagProbability;
            double existingProbability = cell.GetScore(localClosure.parent);
            if (probability > existingProbability){
              // Add it and store backpointer
              added = true;
              cell.BackPointers.put(localClosure.parent, localClosure.child);
              cell.AddUnaryClosure(localClosure.parent, probability);
            }
          }
        }
      }
    }

    private List<PreterminalCell> getPreterminalTags(List<String> sentence) {
      List<PreterminalCell> preterminalCells = new ArrayList<PreterminalCell>();
      int wordIndex = 0;
      for (String word : sentence) {
        HashMap<String, PreterminalTag> preterminalTagsForWord = getPreterminalTags(word);
        preterminalCells.add(new PreterminalCell(preterminalTagsForWord, wordIndex, wordIndex));
        wordIndex++;
      }
      return preterminalCells;
    }

    private HashMap<String, PreterminalTag> getPreterminalTags(String word) {
      HashMap<String, PreterminalTag> preterminalTags = new HashMap<String, PreterminalTag>();
      for (String tag : lexicon.getAllTags()) {
        double score = lexicon.scoreTagging(word, tag);
        if (score > 0) {
          preterminalTags.put(tag, new PreterminalTag(tag, score, word));
        }
      }
      return preterminalTags;
    }

    public CKYParser(List<Tree<String>> trainTrees) {

      System.out.print("Annotating / binarizing training trees ... ");
      List<Tree<String>> annotatedTrainTrees = annotateTrees(trainTrees);
      System.out.println("done.");

      System.out.print("Building grammar ... ");
      grammar = new Grammar(annotatedTrainTrees);
      System.out.println("done. (" + grammar.getStates().size() + " states)");

      // Unary closures is what we'll check to get unary transitions
      unaryClosures = new UnaryClosure(grammar);
      System.out.println(unaryClosures);

      System.out.print("Keeping grammar and setting up a CKY parser ... ");
      lexicon = new Lexicon(annotatedTrainTrees);

      System.out.println("done initializing parser.");
    }

    // This will binarize all trees
    private List<Tree<String>> annotateTrees(List<Tree<String>> trees) {
      List<Tree<String>> annotatedTrees = new ArrayList<Tree<String>>();
      for (Tree<String> tree : trees) {
        annotatedTrees.add(TreeAnnotations.annotateTree(tree));
      }
      return annotatedTrees;
    }
  }

  /**
   * Class which contains code for annotating and binarizing trees for the parser's use, and debinarizing and
   * unannotating them for scoring.
   */
  static class TreeAnnotations {
    public static Tree<String> annotateTree(Tree<String> unAnnotatedTree) {
      // Currently, the only annotation done is a lossless binarization
      // TODO : change the annotation from a lossless binarization to a finite-order markov process (try at least 1st and 2nd order)
      // TODO : mark nodes with the label of their parent nodes, giving a second order vertical markov process
      return binarizeTree(unAnnotatedTree);
    }

    private static Tree<String> binarizeTree(Tree<String> tree) {
      String label = tree.getLabel();
      if (tree.isLeaf())
        return new Tree<String>(label);
      if (tree.getChildren().size() == 1) {
        return new Tree<String>(label, Collections.singletonList(binarizeTree(tree.getChildren().get(0))));
      }
      // otherwise, it's a binary-or-more local tree, so decompose it into a sequence of binary and unary trees.
      String intermediateLabel = "@" + label + "->";
      Tree<String> intermediateTree = binarizeTreeHelper(tree, 0, intermediateLabel);
      return new Tree<String>(label, intermediateTree.getChildren());
    }

    private static Tree<String> binarizeTreeHelper(Tree<String> tree, int numChildrenGenerated, String intermediateLabel) {
      Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
      List<Tree<String>> children = new ArrayList<Tree<String>>();
      children.add(binarizeTree(leftTree));
      if (numChildrenGenerated < tree.getChildren().size() - 1) {
        Tree<String> rightTree = binarizeTreeHelper(tree, numChildrenGenerated + 1, intermediateLabel + "_" + leftTree.getLabel());
        children.add(rightTree);
      }
      return new Tree<String>(intermediateLabel, children);
    }

    public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) {
      // Remove intermediate nodes (labels beginning with "@"
      // Remove all material on node labels which follow their base symbol (cuts at the leftmost -, ^, or : character)
      // Examples: a node with label @NP->DT_JJ will be spliced out, and a node with label NP^S will be reduced to NP
      Tree<String> debinarizedTree = Trees.spliceNodes(annotatedTree, new Filter<String>() {
        public boolean accept(String s) {
          return s.startsWith("@");
        }
      });
      Tree<String> unAnnotatedTree = (new Trees.FunctionNodeStripper()).transformTree(debinarizedTree);
      return unAnnotatedTree;
    }
  }

  /**
   * Simple default implementation of a lexicon, which scores word, tag pairs with a smoothed estimate of
   * P(tag|word)/P(tag).
   */
  static class Lexicon {
    CounterMap<String, String> wordToTagCounters = new CounterMap<String, String>();
    double totalTokens = 0.0;
    double totalWordTypes = 0.0;
    Counter<String> tagCounter = new Counter<String>();
    Counter<String> wordCounter = new Counter<String>();
    Counter<String> typeTagCounter = new Counter<String>();

    public Set<String> getAllTags() {
      return tagCounter.keySet();
    }

    public boolean isKnown(String word) {
      return wordCounter.keySet().contains(word);
    }

    public double scoreTagging(String word, String tag) {
      double p_tag = tagCounter.getCount(tag) / totalTokens;
      double c_word = wordCounter.getCount(word);
      double c_tag_and_word = wordToTagCounters.getCount(word, tag);
      if (c_word < 10) { // rare or unknown
        c_word += 1.0;
        c_tag_and_word += typeTagCounter.getCount(tag) / totalWordTypes;
      }
      double p_word = (1.0 + c_word) / (totalTokens + 1.0);
      double p_tag_given_word = c_tag_and_word / c_word;
      return p_tag_given_word / p_tag * p_word;
    }

    public Lexicon(List<Tree<String>> trainTrees) {
      for (Tree<String> trainTree : trainTrees) {
        List<String> words = trainTree.getYield();
        List<String> tags = trainTree.getPreTerminalYield();
        for (int position = 0; position < words.size(); position++) {
          String word = words.get(position);
          String tag = tags.get(position);
          tallyTagging(word, tag);
        }
      }
    }

    private void tallyTagging(String word, String tag) {
      if (!isKnown(word)) {
        totalWordTypes += 1.0;
        typeTagCounter.incrementCount(tag, 1.0);
      }
      totalTokens += 1.0;
      tagCounter.incrementCount(tag, 1.0);
      wordCounter.incrementCount(word, 1.0);
      wordToTagCounters.incrementCount(word, tag, 1.0);
    }
  }

  /**
   * Simple implementation of a PCFG grammar, offering the ability to look up rules by their child symbols.  Rule
   * probability estimates are just relative frequency estimates off of training trees.
   */
  static class Grammar {
    Map<String, List<BinaryRule>> binaryRulesByLeftChild = new HashMap<String, List<BinaryRule>>();
    Map<String, List<BinaryRule>> binaryRulesByRightChild = new HashMap<String, List<BinaryRule>>();
    Map<String, List<BinaryRule>> binaryRulesByParent = new HashMap<String, List<BinaryRule>>();
    List<BinaryRule> binaryRules = new ArrayList<BinaryRule>();
    Map<String, List<UnaryRule>> unaryRulesByChild = new HashMap<String, List<UnaryRule>>();
    Map<String, List<UnaryRule>> unaryRulesByParent = new HashMap<String, List<UnaryRule>>();
    List<UnaryRule> unaryRules = new ArrayList<UnaryRule>();
    Set<String> states = new HashSet<String>();

    public List<BinaryRule> getBinaryRulesByLeftChild(String leftChild) {
      return CollectionUtils.getValueList(binaryRulesByLeftChild, leftChild);
    }

    public List<BinaryRule> getBinaryRulesByRightChild(String rightChild) {
      return CollectionUtils.getValueList(binaryRulesByRightChild, rightChild);
    }

    public List<BinaryRule> getBinaryRulesByParent(String parent) {
      return CollectionUtils.getValueList(binaryRulesByParent, parent);
    }

    public List<BinaryRule> getBinaryRules() {
      return binaryRules;
    }

    public List<UnaryRule> getUnaryRulesByChild(String child) {
      return CollectionUtils.getValueList(unaryRulesByChild, child);
    }

    public List<UnaryRule> getUnaryRulesByParent(String parent) {
      return CollectionUtils.getValueList(unaryRulesByParent, parent);
    }

    public List<UnaryRule> getUnaryRules() {
      return unaryRules;
    }

    public Set<String> getStates() {
      return states;
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      List<String> ruleStrings = new ArrayList<String>();
      for (String parent : binaryRulesByParent.keySet()) {
        for (BinaryRule binaryRule : getBinaryRulesByParent(parent)) {
          ruleStrings.add(binaryRule.toString());
        }
      }
      for (String parent : unaryRulesByParent.keySet()) {
        for (UnaryRule unaryRule : getUnaryRulesByParent(parent)) {
          ruleStrings.add(unaryRule.toString());
        }
      }
      for (String ruleString : CollectionUtils.sort(ruleStrings)) {
        sb.append(ruleString);
        sb.append("\n");
      }
      return sb.toString();
    }

    private void addBinary(BinaryRule binaryRule) {
      states.add(binaryRule.getParent());
      states.add(binaryRule.getLeftChild());
      states.add(binaryRule.getRightChild());
      binaryRules.add(binaryRule);
      CollectionUtils.addToValueList(binaryRulesByParent, binaryRule.getParent(), binaryRule);
      CollectionUtils.addToValueList(binaryRulesByLeftChild, binaryRule.getLeftChild(), binaryRule);
      CollectionUtils.addToValueList(binaryRulesByRightChild, binaryRule.getRightChild(), binaryRule);
    }

    private void addUnary(UnaryRule unaryRule) {
      states.add(unaryRule.getParent());
      states.add(unaryRule.getChild());
      unaryRules.add(unaryRule);
      CollectionUtils.addToValueList(unaryRulesByParent, unaryRule.getParent(), unaryRule);
      CollectionUtils.addToValueList(unaryRulesByChild, unaryRule.getChild(), unaryRule);
    }

    public Grammar(List<Tree<String>> trainTrees) {
      Counter<UnaryRule> unaryRuleCounter = new Counter<UnaryRule>();
      Counter<BinaryRule> binaryRuleCounter = new Counter<BinaryRule>();
      Counter<String> symbolCounter = new Counter<String>();
      for (Tree<String> trainTree : trainTrees) {
        tallyTree(trainTree, symbolCounter, unaryRuleCounter, binaryRuleCounter);
      }
      for (UnaryRule unaryRule : unaryRuleCounter.keySet()) {
        double unaryProbability = unaryRuleCounter.getCount(unaryRule) / symbolCounter.getCount(unaryRule.getParent());
        unaryRule.setScore(unaryProbability);
        addUnary(unaryRule);
      }
      for (BinaryRule binaryRule : binaryRuleCounter.keySet()) {
        double binaryProbability = binaryRuleCounter.getCount(binaryRule) / symbolCounter.getCount(binaryRule.getParent());
        binaryRule.setScore(binaryProbability);
        addBinary(binaryRule);
      }
    }

    private void tallyTree(Tree<String> tree, Counter<String> symbolCounter, Counter<UnaryRule> unaryRuleCounter, Counter<BinaryRule> binaryRuleCounter) {
      if (tree.isLeaf()) return;
      if (tree.isPreTerminal()) return;
      if (tree.getChildren().size() == 1) {
        UnaryRule unaryRule = makeUnaryRule(tree);
        symbolCounter.incrementCount(tree.getLabel(), 1.0);
        unaryRuleCounter.incrementCount(unaryRule, 1.0);
      }
      if (tree.getChildren().size() == 2) {
        BinaryRule binaryRule = makeBinaryRule(tree);
        symbolCounter.incrementCount(tree.getLabel(), 1.0);
        binaryRuleCounter.incrementCount(binaryRule, 1.0);
      }
      if (tree.getChildren().size() < 1 || tree.getChildren().size() > 2) {
        throw new RuntimeException("Attempted to construct a Grammar with an illegal tree (unbinarized?): " + tree);
      }
      for (Tree<String> child : tree.getChildren()) {
        tallyTree(child, symbolCounter, unaryRuleCounter, binaryRuleCounter);
      }
    }

    private UnaryRule makeUnaryRule(Tree<String> tree) {
      return new UnaryRule(tree.getLabel(), tree.getChildren().get(0).getLabel());
    }

    private BinaryRule makeBinaryRule(Tree<String> tree) {
      return new BinaryRule(tree.getLabel(), tree.getChildren().get(0).getLabel(), tree.getChildren().get(1).getLabel());
    }
  }

  static class BinaryRule {
    String parent;
    String leftChild;
    String rightChild;
    double score;

    public String getParent() {
      return parent;
    }

    public String getLeftChild() {
      return leftChild;
    }

    public String getRightChild() {
      return rightChild;
    }

    public double getScore() {
      return score;
    }

    public void setScore(double score) {
      this.score = score;
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof BinaryRule)) return false;

      final BinaryRule binaryRule = (BinaryRule) o;

      if (leftChild != null ? !leftChild.equals(binaryRule.leftChild) : binaryRule.leftChild != null) return false;
      if (parent != null ? !parent.equals(binaryRule.parent) : binaryRule.parent != null) return false;
      if (rightChild != null ? !rightChild.equals(binaryRule.rightChild) : binaryRule.rightChild != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = (parent != null ? parent.hashCode() : 0);
      result = 29 * result + (leftChild != null ? leftChild.hashCode() : 0);
      result = 29 * result + (rightChild != null ? rightChild.hashCode() : 0);
      return result;
    }

    public String toString() {
      return parent + " -> " + leftChild + " " + rightChild + " %% " + score;
    }

    public BinaryRule(String parent, String leftChild, String rightChild) {
      this.parent = parent;
      this.leftChild = leftChild;
      this.rightChild = rightChild;
    }
  }

  static class UnaryRule {
    String parent;
    String child;
    double score;

    public String getParent() {
      return parent;
    }

    public String getChild() {
      return child;
    }

    public double getScore() {
      return score;
    }

    public void setScore(double score) {
      this.score = score;
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof UnaryRule)) return false;

      final UnaryRule unaryRule = (UnaryRule) o;

      if (child != null ? !child.equals(unaryRule.child) : unaryRule.child != null) return false;
      if (parent != null ? !parent.equals(unaryRule.parent) : unaryRule.parent != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = (parent != null ? parent.hashCode() : 0);
      result = 29 * result + (child != null ? child.hashCode() : 0);
      return result;
    }

    public String toString() {
      return parent + " -> " + child + " %% " + score;
    }

    public UnaryRule(String parent, String child) {
      this.parent = parent;
      this.child = child;
    }
  }

  /**
   * Calculates and provides accessors for the REFLEXIVE, TRANSITIVE closure of the unary rules in the provided Grammar.
   * Each rule in this closure stands for zero or more unary rules in the original grammar.  Use the getPath() method to
   * retrieve the full sequence of symbols (from parent to child) which support that path.
   */
  static class UnaryClosure {
    Map<String, List<UnaryRule>> closedUnaryRulesByChild = new HashMap<String, List<UnaryRule>>();
    Map<String, List<UnaryRule>> closedUnaryRulesByParent = new HashMap<String, List<UnaryRule>>();
    Map<UnaryRule, List<String>> pathMap = new HashMap<UnaryRule, List<String>>();

    public List<UnaryRule> getClosedUnaryRulesByChild(String child) {
      return CollectionUtils.getValueList(closedUnaryRulesByChild, child);
    }

    public List<UnaryRule> getClosedUnaryRulesByParent(String parent) {
      return CollectionUtils.getValueList(closedUnaryRulesByParent, parent);
    }

    public List<String> getPath(UnaryRule unaryRule) {
      return pathMap.get(unaryRule);
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      for (String parent : closedUnaryRulesByParent.keySet()) {
        for (UnaryRule unaryRule : getClosedUnaryRulesByParent(parent)) {
          List<String> path = getPath(unaryRule);
//          if (path.size() == 2) continue;
          sb.append(unaryRule);
          sb.append("  ");
          sb.append(path);
          sb.append("\n");
        }
      }
      return sb.toString();
    }

    public UnaryClosure(Collection<UnaryRule> unaryRules) {
      Map<UnaryRule, List<String>> closureMap = computeUnaryClosure(unaryRules);
      for (UnaryRule unaryRule : closureMap.keySet()) {
        addUnary(unaryRule, closureMap.get(unaryRule));
      }
    }

    public UnaryClosure(Grammar grammar) {
      this(grammar.getUnaryRules());
    }

    private void addUnary(UnaryRule unaryRule, List<String> path) {
      CollectionUtils.addToValueList(closedUnaryRulesByChild, unaryRule.getChild(), unaryRule);
      CollectionUtils.addToValueList(closedUnaryRulesByParent, unaryRule.getParent(), unaryRule);
      pathMap.put(unaryRule, path);
    }

    private static Map<UnaryRule, List<String>> computeUnaryClosure(Collection<UnaryRule> unaryRules) {

      Map<UnaryRule, String> intermediateStates = new HashMap<UnaryRule, String>();
      Counter<UnaryRule> pathCosts = new Counter<UnaryRule>();
      Map<String, List<UnaryRule>> closedUnaryRulesByChild = new HashMap<String, List<UnaryRule>>();
      Map<String, List<UnaryRule>> closedUnaryRulesByParent = new HashMap<String, List<UnaryRule>>();

      Set<String> states = new HashSet<String>();

      for (UnaryRule unaryRule : unaryRules) {
        relax(pathCosts, intermediateStates, closedUnaryRulesByChild, closedUnaryRulesByParent, unaryRule, null, unaryRule.getScore());
        states.add(unaryRule.getParent());
        states.add(unaryRule.getChild());
      }

      for (String intermediateState : states) {
        List<UnaryRule> incomingRules = closedUnaryRulesByChild.get(intermediateState);
        List<UnaryRule> outgoingRules = closedUnaryRulesByParent.get(intermediateState);
        if (incomingRules == null || outgoingRules == null) continue;
        for (UnaryRule incomingRule : incomingRules) {
          for (UnaryRule outgoingRule : outgoingRules) {
            UnaryRule rule = new UnaryRule(incomingRule.getParent(), outgoingRule.getChild());
            double newScore = pathCosts.getCount(incomingRule) * pathCosts.getCount(outgoingRule);
            relax(pathCosts, intermediateStates, closedUnaryRulesByChild, closedUnaryRulesByParent, rule, intermediateState, newScore);
          }
        }
      }

      for (String state : states) {
        UnaryRule selfLoopRule = new UnaryRule(state, state);
        relax(pathCosts, intermediateStates, closedUnaryRulesByChild, closedUnaryRulesByParent, selfLoopRule, null, 1.0);
      }

      Map<UnaryRule, List<String>> closureMap = new HashMap<UnaryRule, List<String>>();

      for (UnaryRule unaryRule : pathCosts.keySet()) {
        unaryRule.setScore(pathCosts.getCount(unaryRule));
        List<String> path = extractPath(unaryRule, intermediateStates);
        closureMap.put(unaryRule, path);
      }

      System.out.println("SIZE: " + closureMap.keySet().size());

      return closureMap;

    }

    private static List<String> extractPath(UnaryRule unaryRule, Map<UnaryRule, String> intermediateStates) {
      List<String> path = new ArrayList<String>();
      path.add(unaryRule.getParent());
      String intermediateState = intermediateStates.get(unaryRule);
      if (intermediateState != null) {
        List<String> parentPath = extractPath(new UnaryRule(unaryRule.getParent(), intermediateState), intermediateStates);
        for (int i = 1; i < parentPath.size() - 1; i++) {
          String state = parentPath.get(i);
          path.add(state);
        }
        path.add(intermediateState);
        List<String> childPath = extractPath(new UnaryRule(intermediateState, unaryRule.getChild()), intermediateStates);
        for (int i = 1; i < childPath.size() - 1; i++) {
          String state = childPath.get(i);
          path.add(state);
        }
      }
      if (path.size() == 1 && unaryRule.getParent().equals(unaryRule.getChild())) return path;
      path.add(unaryRule.getChild());
      return path;
    }


    private static void relax(Counter<UnaryRule> pathCosts, Map<UnaryRule, String> intermediateStates, Map<String, List<UnaryRule>> closedUnaryRulesByChild, Map<String, List<UnaryRule>> closedUnaryRulesByParent, UnaryRule unaryRule, String intermediateState, double newScore) {
      if (intermediateState != null && (intermediateState.equals(unaryRule.getParent()) || intermediateState.equals(unaryRule.getChild()))) return;
      boolean isNewRule = !pathCosts.containsKey(unaryRule);
      double oldScore = (isNewRule ? Double.NEGATIVE_INFINITY : pathCosts.getCount(unaryRule));
      if (oldScore > newScore) return;
      if (isNewRule) {
        CollectionUtils.addToValueList(closedUnaryRulesByChild, unaryRule.getChild(), unaryRule);
        CollectionUtils.addToValueList(closedUnaryRulesByParent, unaryRule.getParent(), unaryRule);
      }
      pathCosts.setCount(unaryRule, newScore);
      intermediateStates.put(unaryRule, intermediateState);
    }

  }


  public static void main(String[] args) {
    // Parse command line flags and arguments
    Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);

    // Set up default parameters and settings
    String basePath = ".";
    boolean verbose = true;
    String testMode = "validate";
    int maxTrainLength = 1000;
    int maxTestLength = 40;

    // Update defaults using command line specifications
    if (argMap.containsKey("-path")) {
      basePath = argMap.get("-path");
      System.out.println("Using base path: " + basePath);
    }
    if (argMap.containsKey("-test")) {
      testMode = "test";
      System.out.println("Testing on final test data.");
    } else {
      System.out.println("Testing on validation data.");
    }
    if (argMap.containsKey("-maxTrainLength")) {
      maxTrainLength = Integer.parseInt(argMap.get("-maxTrainLength"));
    }
    System.out.println("Maximum length for training sentences: " + maxTrainLength);
    if (argMap.containsKey("-maxTestLength")) {
      maxTestLength = Integer.parseInt(argMap.get("-maxTestLength"));
    }
    System.out.println("Maximum length for test sentences: " + maxTestLength);
    if (argMap.containsKey("-verbose")) {
      verbose = true;
    }
    if (argMap.containsKey("-quiet")) {
      verbose = false;
    }

    System.out.print("Loading training trees (sections 2-21) ... ");
    List<Tree<String>> trainTrees = readTrees(basePath, 200, 2199, maxTrainLength);
    System.out.println("done. (" + trainTrees.size() + " trees)");
    List<Tree<String>> testTrees = null;
    if (testMode.equalsIgnoreCase("validate")) {
      System.out.print("Loading validation trees (section 22) ... ");
      testTrees = readTrees(basePath, 2200, 2299, maxTestLength);
    } else {
      System.out.print("Loading test trees (section 23) ... ");
      testTrees = readTrees(basePath, 2300, 2399, maxTestLength);
    }
    System.out.println("done. (" + testTrees.size() + " trees)");

    Parser parser = new CKYParser(trainTrees);

    testParser(parser, testTrees, verbose);
  }

  private static void testParser(Parser parser, List<Tree<String>> testTrees, boolean verbose) {
    EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> eval = new EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>(Collections.singleton("ROOT"), new HashSet<String>(Arrays.asList(new String[]{"''", "``", ".", ":", ","})));
    for (Tree<String> testTree : testTrees) {
      List<String> testSentence = testTree.getYield();
      Tree<String> guessedTree = parser.getBestParse(testSentence);
      if (verbose) {
        System.out.println("Guess:\n" + Trees.PennTreeRenderer.render(guessedTree));
        System.out.println("Gold:\n" + Trees.PennTreeRenderer.render(testTree));
      }
      eval.evaluate(guessedTree, testTree);
    }
    eval.display(true);
  }

  private static List<Tree<String>> readTrees(String basePath, int low, int high, int maxLength) {
    Collection<Tree<String>> trees = PennTreebankReader.readTrees(basePath, low, high);
    // normalize trees
    Trees.TreeTransformer<String> treeTransformer = new Trees.StandardTreeNormalizer();
    List<Tree<String>> normalizedTreeList = new ArrayList<Tree<String>>();
    for (Tree<String> tree : trees) {
      Tree<String> normalizedTree = treeTransformer.transformTree(tree);
      if (normalizedTree.getYield().size() > maxLength)
        continue;
//      System.out.println(Trees.PennTreeRenderer.render(normalizedTree));
      normalizedTreeList.add(normalizedTree);
    }
    return normalizedTreeList;
  }
}
