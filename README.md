## Learnings

A torch.fx.GraphModule is an nn.Module generated from a torch.fx.Graph. Graph modules
have code.

SESE regions form a **laminar family**


**Goal:**  
Given a set of SESE regions (each with `entry`, `exit`, and a heuristic `score`), produce a laminar subset — meaning any two chosen regions are either disjoint or nested, never partially overlapping.

---

**Algorithm outline**

1. **Assign intervals**  
For each region `r`, compute:  
`interval(r) = [tin(entry(r)), tout(exit(r))]`  
where `tin` and `tout` come from a DFS traversal of the dominator tree.  
Two regions are disjoint if one interval ends before the other starts.

---

2. **Sort regions**  
Sort by descending score (best first), and then by start time:  
`regions.sort(key=lambda r: (-score(r), r.interval.start))`

---

3. **Greedy laminarization**  
Iterate through sorted regions, keeping only those that do not partially overlap an already accepted region.  
If two regions intersect but neither contains the other, drop the one with lower score.

Steps:
- Initialize an empty list of accepted regions.
- For each region in order:
  - Compare it to previously accepted ones.
  - If it partially overlaps with any existing region (not nested or disjoint):
    - Keep only the higher-scored region.
  - Otherwise, accept it.
- Continue until all are processed.

Definitions:
- Two regions overlap if their intervals intersect but neither fully contains the other.
- Containment means one interval completely includes the other.

---

4. **Optional structural checks**  
After building the laminar set, verify:
- The parent’s entry dominates the child’s entry.
- The child’s exit postdominates the parent’s exit.  
If not, discard the inconsistent region.

---

**Result:**  
A laminar (nested-or-disjoint) subset of SESE regions, filtered according to your heuristic priorities.  
This yields a consistent hierarchy suitable for region trees or safe parallelization.

**Goal:**  
Construct a *region tree* from a control-flow graph (CFG) using SESE (Single-Entry Single-Exit) regions.  
The resulting tree captures the hierarchical, nested structure of regions: each node is a SESE region whose children are the maximal subregions contained within it.

---

**Algorithm outline**

1. **Identify SESE regions**  
Use dominance and post-dominance information:
- A region `R` is SESE if it has:
  - A unique `entry` node `E` such that `E` dominates all nodes in `R`.
  - A unique `exit` node `X` such that `X` postdominates all nodes in `R`.
- For reducible CFGs, each edge `(E, X)` that satisfies these conditions defines one SESE region.

---

2. **Compute structural intervals**  
Assign each node in the CFG two timestamps:
- `tin(n)` — preorder index from a DFS on the dominator tree.
- `tout(n)` — postorder index from the same DFS.  
For each region `(E, X)`, record the interval `[tin(E), tout(X)]`.

These intervals allow you to test **containment** (`A` contains `B` if A’s interval fully covers B’s) and **disjointness**.

---

3. **Sort regions by structural nesting**  
Sort all SESE regions by ascending `tin(entry)` and descending `tout(exit)`.  
This guarantees that outer (larger) regions appear before their subregions.

---

4. **Build the tree**  
Iterate over sorted regions and insert each into the appropriate parent:
- Maintain a stack of active regions representing the current nesting path.
- For each region `R`:
  - While the stack’s top region does **not** contain `R`, pop it.
  - If the stack is not empty, the top region is `R`’s parent.
  - Insert `R` as a child of that parent.
  - Push `R` onto the stack.

At the end, the tree’s root will represent the entire procedure or CFG (the outermost SESE region).

---

5. **Optional consistency verification**  
After construction, confirm:
- Each child’s `entry` is dominated by its parent’s `entry`.
- Each parent’s `exit` postdominates its child’s `exit`.
- No two siblings overlap.

---

**Result:**  
A **region tree** where:
- Each node corresponds to a SESE region.
- Containment represents structural nesting.
- Children are the maximal SESE subregions within a parent.  
This hierarchical structure is laminar and can be used for structured analysis, transformation, or parallelization.