---
name: grill-me
description: Interview the user relentlessly about a Deep Reader plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when the user wants to stress-test a plan, get grilled on their design, or when Main Agent V2 requests a pre-implementation challenge.
---

# Grill Me

Interview the maintainer relentlessly about every aspect of this plan until reaching shared understanding.

Walk down each branch of the design tree.

Resolve dependencies between decisions one by one.

Ask the questions one at a time.

For each question:

1. ask the question
2. explain why this question matters
3. provide your recommended answer
4. state what repository evidence could answer it

If a question can be answered by exploring the codebase or repository markdown, explore the repository instead of asking the maintainer.

This skill is read-only.

Do not modify files.

Do not implement code.

Do not update markdown.

# Interaction Rule

This skill is an interview skill.

It must ask the maintainer questions.

It must not answer its own questions.

It must not produce a full challenge report before the maintainer has answered.

Ask exactly one question at a time.

After asking a question, stop and wait for the maintainer's answer.

For each question:

1. ask one question
2. explain why it matters
3. explain what kind of answer is needed
4. stop

Do not continue until the maintainer answers.

## Deep Reader Required Context

When used for Deep Reader, inspect relevant repository memory first:

- AGENTS.md
- Deep_Reflective_Reader/proposal.md
- Deep_Reflective_Reader/high-level-design.md
- Deep_Reflective_Reader/docs/modules/index.md
- target module module-detailed-design.md
- target module module-checklist.md
- Deep_Reflective_Reader/progress.md

## Challenge Focus

Challenge:

- ownership correctness
- module boundary
- hidden scope expansion
- architecture assumptions
- missing repository evidence
- unclear persistence semantics
- checklist/progress impact
- testing requirements
- rollback risk

## Stop Condition

Stop when:

- the plan is clear enough to implement
- the maintainer rejects the plan
- ownership is unresolved
- repository memory conflicts with the plan
- the task should be split

## Final Output

After the questioning loop, report:

```text
Skill:
grill-me

Target Module:
...

Plan Under Review:
...

Resolved Decisions:
- ...

Unresolved Decisions:
- ...

Repository Evidence Used:
- ...

Recommended Action:
proceed / revise / split / stop

Reason:
...