"""Source text for the mock policy corpus.

Cobalt Ridge Financial is a fictional institution. Figures are illustrative;
several mirror public US rules (for example the $10,000 Currency Transaction
Report threshold and Regulation E error-resolution timelines) so that the mock
policies read realistically next to the public regulatory documents.

Each document is a list of pages. Within a page, lines starting with ``# `` are
titles, ``## `` are section headings, and blank lines separate paragraphs.
"""

INSTITUTION = "Cobalt Ridge Financial (fictional institution for demonstration purposes)"

DOCUMENTS: dict[str, list[str]] = {
    "transaction_limits_policy.pdf": [
        """# Transaction Limits and Authorization Policy
Policy ID: CRF-OPS-101. Version 4.2. Owner: Payments Risk Committee. Effective: January 15, 2026.

## 1. Purpose and Scope
This policy sets the default limits that apply to consumer and business accounts for card, ATM, ACH and wire transactions, and the authorization controls required when a transaction exceeds those limits. It applies to all channels: branch, online banking, mobile and contact center.

## 2. Consumer Card and ATM Limits
2.1 The default daily ATM cash withdrawal limit for consumer debit cards is $1,000 per card per calendar day.
2.2 The default daily point-of-sale purchase limit for consumer debit cards is $5,000 per card per calendar day.
2.3 Premier tier customers have a daily ATM withdrawal limit of $2,500 and a daily purchase limit of $10,000.
2.4 A customer may request a temporary limit increase of up to 50 percent above the default for a maximum of 7 calendar days. Temporary increases require identity verification through a one-time passcode.

## 3. Card Velocity Controls
3.1 More than 5 card-not-present transactions within 10 minutes on the same card automatically triggers a fraud review and a temporary hold on further card-not-present activity.
3.2 Three consecutive declined PIN attempts lock the card for 24 hours.""",
        """## 4. Wire Transfer Limits
4.1 Consumer accounts may send outgoing domestic wires up to $25,000 per business day through online banking.
4.2 Business accounts may send outgoing wires up to $250,000 per business day through online banking, subject to the entitlements configured by the business administrator.
4.3 International wires from consumer accounts are limited to $10,000 per business day through digital channels. Larger international wires must be initiated in branch.
4.4 Any single wire transfer of $50,000 or more requires dual authorization: the initiator and a second authorized approver who is not the initiator.
4.5 Wire requests received after the 4:00 p.m. Eastern Time cutoff are processed on the next business day.

## 5. ACH Limits
5.1 Consumer outgoing ACH transfers are limited to $15,000 per business day and $40,000 per rolling 30-day period.
5.2 Same-day ACH is available for individual payments up to $1,000,000, in line with the network per-payment limit.
5.3 New accounts open for fewer than 30 days are limited to $2,500 per day in outgoing ACH transfers.""",
        """## 6. Exceptions and Overrides
6.1 Limit exceptions may be approved by a Branch Manager up to 2 times the default limit, and by the Payments Risk Committee above that.
6.2 Every override must record the business justification, the approver, and the expiry date in the case management system.
6.3 Overrides may not be granted to accounts with an open fraud case or an unresolved suspicious activity alert.

## 7. Monitoring and Review
7.1 Payments Risk reviews limit utilization quarterly and recommends changes to the Payments Risk Committee.
7.2 This policy is reviewed at least annually. Material changes require approval by the Chief Risk Officer.""",
    ],
    "aml_kyc_policy.pdf": [
        """# Anti-Money Laundering and Know Your Customer Policy
Policy ID: CRF-CMP-210. Version 7.0. Owner: BSA/AML Officer. Effective: March 1, 2026.

## 1. Purpose
This policy establishes the controls Cobalt Ridge Financial uses to comply with the Bank Secrecy Act, USA PATRIOT Act and OFAC sanctions requirements, and to detect and report money laundering and terrorist financing.

## 2. Customer Identification Program (CIP)
2.1 Before opening an account, staff must collect at minimum the customer's full legal name, date of birth, residential address and an identification number (a taxpayer identification number for US persons).
2.2 Identity must be verified through documentary methods (an unexpired government-issued photo ID) or non-documentary methods (such as a consumer reporting agency check) within 5 business days of account opening.
2.3 Accounts for which identity cannot be verified within that period must be restricted and then closed.

## 3. Customer Due Diligence and Beneficial Ownership
3.1 For legal entity customers, staff must identify each individual who owns 25 percent or more of the equity interests, and one individual with significant responsibility to control the entity.
3.2 Each customer is assigned a risk rating of Low, Medium or High at onboarding.""",
        """## 4. Enhanced Due Diligence (EDD)
4.1 Enhanced due diligence is mandatory for High-risk customers, including politically exposed persons (PEPs), money services businesses, and customers with significant activity in high-risk jurisdictions.
4.2 EDD requires documentation of source of funds and source of wealth, and approval by the BSA/AML Officer before the relationship is opened.
4.3 High-risk customers are re-reviewed every 12 months, Medium-risk customers every 24 months, and Low-risk customers every 36 months.

## 5. Currency Transaction Reports (CTR)
5.1 A Currency Transaction Report must be filed with FinCEN for each transaction in currency of more than $10,000 conducted by, through or to the institution in a single business day.
5.2 Multiple cash transactions by or on behalf of the same person must be aggregated when they total more than $10,000 in one business day.
5.3 CTRs must be filed within 15 calendar days after the date of the transaction.

## 6. Structuring
Structuring means breaking up cash transactions to evade CTR reporting, for example making several deposits of $9,500 on consecutive days. Structuring is a federal crime regardless of the source of funds. Staff must never advise customers on how to avoid reporting thresholds. Suspected structuring must be escalated to the Financial Intelligence Unit (FIU) the same day.""",
        """## 7. Suspicious Activity Reports (SAR)
7.1 A SAR must be filed for suspicious transactions aggregating $5,000 or more where a suspect can be identified.
7.2 A SAR must be filed no later than 30 calendar days after the date of initial detection of facts that may constitute a basis for filing. If no suspect is identified, filing may be delayed an additional 30 calendar days, but no more than 60 calendar days after initial detection.
7.3 Continuing activity reviews are performed every 90 days after a SAR is filed.
7.4 SARs and the fact that a SAR was filed are confidential. Disclosing a SAR to the subject of the report ("tipping off") is prohibited.

## 8. Sanctions Screening
All customers, beneficial owners and wire counterparties are screened against the OFAC Specially Designated Nationals list at onboarding and daily thereafter. Potential matches must be dispositioned within 24 hours.

## 9. Record Retention and Training
9.1 CIP records, CTRs, SARs and supporting documentation are retained for 5 years.
9.2 All customer-facing employees complete AML training within 30 days of hire and annually thereafter.""",
    ],
    "fee_schedule.pdf": [
        """# Consumer Fee Schedule
Document ID: CRF-PRD-330. Effective: February 1, 2026. Applies to consumer checking, savings and credit card products.

## 1. Checking Account Fees
1.1 Monthly maintenance fee: $12. Waived when the average daily balance is $1,500 or more, or when monthly direct deposits total $500 or more.
1.2 Overdraft fee: $34 per item, with a maximum of 3 overdraft fees per business day. No overdraft fee is charged when the account is overdrawn by $50 or less at the end of the business day.
1.3 Returned item (non-sufficient funds) fee: $0. Cobalt Ridge Financial does not charge NSF fees.
1.4 Stop payment fee: $30 per request.
1.5 Out-of-network ATM fee: $2.50 per withdrawal. Premier tier customers receive up to 4 out-of-network ATM fee refunds per statement cycle.

## 2. Wire Transfer Fees
2.1 Outgoing domestic wire: $25.
2.2 Outgoing international wire: $45, plus any intermediary bank charges.
2.3 Incoming wire: $15. Waived for Premier tier customers.""",
        """## 3. Credit Card Fees
3.1 Annual fee: $95 for the Cobalt Rewards card, waived for the first year.
3.2 Late payment fee: up to $30 for the first late payment, and up to $41 for each additional late payment within the next six billing cycles.
3.3 Returned payment fee: up to $30.
3.4 Cash advance fee: 5 percent of the amount of each cash advance, or $10, whichever is greater.
3.5 Balance transfer fee: 3 percent of the amount of each transfer, or $5, whichever is greater.
3.6 Foreign transaction fee: 2.7 percent of each transaction made in a foreign currency or processed outside the United States, after conversion to US dollars.

## 4. Other Services
4.1 Expedited card replacement: $25.
4.2 Cashier's check: $10.
4.3 Paper statement fee: $2 per statement cycle. Waived for customers aged 65 or older.

## 5. Fee Reversals
Customer-facing staff may reverse up to $100 in fees per customer per calendar year without manager approval. Reversals above that amount require Branch Manager approval and must be documented with the reason.""",
    ],
    "dispute_resolution_procedure.txt": [
        """# Card and Electronic Transfer Dispute Resolution Procedure
Procedure ID: CRF-OPS-415. Version 3.1. Owner: Disputes Operations. Effective: April 1, 2026.

## 1. Scope
This procedure covers disputes of unauthorized or incorrect electronic fund transfers on consumer accounts, including debit card, ATM, ACH and online transfers, and billing errors on consumer credit cards.

## 2. Customer Notice Deadlines
2.1 Customers must report an error on an electronic fund transfer within 60 calendar days after the statement on which the error first appeared was sent.
2.2 Credit card billing errors must be reported in writing within 60 days after the statement containing the error was sent.

## 3. Consumer Liability for Unauthorized Debit Transfers
3.1 If the customer reports a lost or stolen debit card within 2 business days of learning of it, liability is limited to the lesser of $50 or the amount of unauthorized transfers.
3.2 If the customer reports after 2 business days but within 60 days of the statement, liability is limited to $500.
3.3 If the customer fails to report within 60 days of the statement, the customer may be liable for unauthorized transfers that occur after the 60-day period and before notice is given.
3.4 Under the Cobalt Ridge Zero Liability promise, customers who report promptly are not held liable for unauthorized card transactions, as long as the card was not shared and the customer did not participate in the transaction.""",
        """## 4. Investigation Timelines
4.1 Disputes Operations must acknowledge each dispute within 1 business day.
4.2 Investigations must be completed within 10 business days after notice of the error.
4.3 If the investigation cannot be completed within 10 business days, provisional credit for the disputed amount must be issued within 10 business days, and the investigation may take up to 45 calendar days.
4.4 For transfers initiated outside the United States, point-of-sale debit card transactions, and accounts open fewer than 30 days, the investigation may take up to 90 calendar days.
4.5 The customer must be notified of the results within 3 business days after the investigation is completed. If provisional credit is reversed, the customer receives 5 business days of notice before the reversal.

## 5. Chargebacks to Merchants
5.1 Card disputes are submitted to the card network within 30 days of the customer's notice.
5.2 Disputes Operations must keep supporting documents for 2 years after the case closes.

## 6. Escalation
Disputes over $10,000, disputes involving suspected employee fraud, and repeat disputes on the same account within 90 days must be escalated to the Disputes Operations manager.""",
    ],
    "customer_data_privacy_policy.txt": [
        """# Customer Data Privacy and Protection Policy
Policy ID: CRF-SEC-120. Version 5.3. Owner: Chief Privacy Officer. Effective: January 1, 2026.

## 1. Data Classification
1.1 Restricted data includes Social Security numbers, full account and card numbers, authentication credentials, and biometric data.
1.2 Confidential data includes customer names combined with balances, transaction history, and contact details.
1.3 Internal data includes policies, procedures and non-customer operational reports.

## 2. Handling Requirements
2.1 Restricted data must be encrypted at rest using AES-256 and in transit using TLS 1.2 or higher.
2.2 Only the last 4 digits of an account or card number may be displayed in customer correspondence, screens shared with customers, and internal reports.
2.3 Restricted data must never be entered into external tools, including public generative AI services. Approved internal AI tools must mask personal data before any model inference.
2.4 Restricted data may not be sent by email unless the message is encrypted.""",
        """## 3. Access Control
3.1 Access to restricted data follows least privilege and requires manager approval.
3.2 User access to systems holding restricted data is recertified every quarter.
3.3 Privileged access sessions are logged and the logs are retained for 1 year.

## 4. Retention and Disposal
4.1 Account records are retained for 7 years after the account is closed, unless a longer period is required by law or a legal hold.
4.2 Paper records containing restricted data must be shredded. Electronic media must be securely wiped or physically destroyed.

## 5. Incident Response and Breach Notification
5.1 Employees must report any suspected data incident to the Security Operations Center within 1 hour of discovery.
5.2 The Chief Privacy Officer determines whether notification is required. Where required, affected customers are notified without unreasonable delay and no later than 30 days after the breach is confirmed.
5.3 The primary federal regulator is notified within 36 hours of determining that a notification incident has occurred.

## 6. Customer Rights
Customers may opt out of sharing their information with non-affiliated third parties for marketing at any time. Opt-out requests are processed within 30 days.""",
    ],
    "ai_model_governance_policy.pdf": [
        """# AI and Model Risk Governance Policy
Policy ID: CRF-RSK-505. Version 2.0. Owner: Model Risk Management. Effective: May 1, 2026.

## 1. Purpose and Scope
This policy governs the development, validation, deployment and monitoring of models and AI systems at Cobalt Ridge Financial, including generative AI assistants used by employees. It aligns with supervisory guidance on model risk management.

## 2. Model Inventory and Risk Tiering
2.1 Every model and AI system must be registered in the model inventory before use in production.
2.2 Models are assigned a risk tier from Tier 1 (highest) to Tier 3 (lowest), based on financial impact, customer impact and regulatory exposure.
2.3 Tier 1 models must be independently validated before deployment and revalidated every 12 months. Tier 2 models are revalidated every 24 months and Tier 3 models every 36 months.

## 3. Generative AI Requirements
3.1 Generative AI assistants must ground answers in approved internal sources and display citations to the source document.
3.2 Personal data must be masked before any prompt is sent to a model, whether hosted internally or by a vendor.
3.3 Generative AI may not make or communicate final credit decisions, adverse action notices, or SAR filing decisions. A qualified human must review and approve these outcomes.""",
        """## 4. Logging and Explainability
4.1 Each generative AI interaction must be logged with the user query, masked query, retrieved sources, model prompt, model response and response latency.
4.2 Interaction logs are retained for 3 years and are available to Internal Audit and Model Risk Management.

## 5. Monitoring
5.1 Production AI systems are monitored monthly for accuracy, hallucination rate, latency and data leakage.
5.2 A hallucination rate above 2 percent on the monthly evaluation set, or any confirmed leakage of restricted data, triggers an incident review and may require the system to be suspended.
5.3 Vendor-hosted models require a third-party risk assessment before onboarding and an annual review thereafter.

## 6. Roles
The Model Risk Management team owns this policy. Model owners are accountable for documentation and monitoring. The AI Governance Council approves Tier 1 generative AI use cases.""",
    ],
}
