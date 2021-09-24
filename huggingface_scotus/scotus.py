from dataclasses import dataclass
import datasets
from datasets.info import DatasetInfo
from datasets.load import load_dataset
from datasets.utils.download_manager import DownloadManager
import os
import jsonlines
import json

_DESCRIPTION = """Dataset extracted from case laws of Supreme Court of United States."""

_VERSION = "0.4.0"
_DOWNLOAD_URL='https://sid.erda.dk/share_redirect/ceCWid0G3T'
ISSUE_AREAS = "Criminal Procedure,Civil Rights,First Amendment,Due Process,Privacy,Attorneys,Unions,Economic Activity,Judicial Power,Federalism,Federal Taxation".split(
    ","
)

PERSON = "person"
PUBLIC_ENTITY = "public_entity"
ORGANIZATION = "organization"
OTHER = "other"
FACILITY = "facility"

RESPONDENT_MAPPING = {
    "attorney general of the United States, or his office": PERSON,
    "specified state board or department of education": PUBLIC_ENTITY,
    "city, town, township, village, or borough government or governmental unit": PUBLIC_ENTITY,
    "state commission, board, committee, or authority": PUBLIC_ENTITY,
    "county government or county governmental unit, except school district": PUBLIC_ENTITY,
    "court or judicial district": PUBLIC_ENTITY,
    "state department or agency": PUBLIC_ENTITY,
    "governmental employee or job applicant": PERSON,
    "female governmental employee or job applicant": PERSON,
    "minority governmental employee or job applicant": PERSON,
    "minority female governmental employee or job applicant": PERSON,
    "not listed among agencies in the first Administrative Action variable": OTHER,
    "retired or former governmental employee": PERSON,
    "U.S. House of Representatives": PUBLIC_ENTITY,
    "interstate compact": PUBLIC_ENTITY,
    "judge": PERSON,
    "state legislature, house, or committee": PUBLIC_ENTITY,
    "local governmental unit other than a county, city, town, township, village, or borough": PUBLIC_ENTITY,
    "governmental official, or an official of an agency established under an interstate compact": PUBLIC_ENTITY,
    "state or U.S. supreme court": PUBLIC_ENTITY,
    "local school district or board of education": PUBLIC_ENTITY,
    "U.S. Senate": PUBLIC_ENTITY,
    "U.S. senator": PERSON,
    "foreign nation or instrumentality": PUBLIC_ENTITY,
    "state or local governmental taxpayer, or executor of the estate of": PUBLIC_ENTITY,
    "state college or university": PUBLIC_ENTITY,
    "United States": PUBLIC_ENTITY,
    "State": PUBLIC_ENTITY,
    "person accused, indicted, or suspected of crime": PERSON,
    "advertising business or agency": ORGANIZATION,
    "agent, fiduciary, trustee, or executor": ORGANIZATION,
    "airplane manufacturer, or manufacturer of parts of airplanes": ORGANIZATION,
    "airline": ORGANIZATION,
    "distributor, importer, or exporter of alcoholic beverages": ORGANIZATION,
    "alien, person subject to a denaturalization proceeding, or one whose citizenship is revoked": PERSON,
    "American Medical Association": PERSON,
    "National Railroad Passenger Corp.": ORGANIZATION,
    "amusement establishment, or recreational facility": FACILITY,
    "arrested person, or pretrial detainee": PERSON,
    "attorney, or person acting as such;includes bar applicant or law student, or law firm or bar association": PERSON,
    "author, copyright holder": OTHER,
    "bank, savings and loan, credit union, investment company": ORGANIZATION,
    "bankrupt person or business, or business in reorganization": OTHER,
    "establishment serving liquor by the glass, or package liquor store": ORGANIZATION,
    "water transportation, stevedore": OTHER,
    "bookstore, newsstand, printer, bindery, purveyor or distributor of books or magazines": ORGANIZATION,
    "brewery, distillery": ORGANIZATION,
    "broker, stock exchange, investment or securities firm": ORGANIZATION,
    "construction industry": ORGANIZATION,
    "bus or motorized passenger transportation vehicle": OTHER,
    "business, corporation": ORGANIZATION,
    "buyer, purchaser": OTHER,
    "cable TV": OTHER,
    "car dealer": PERSON,
    "person convicted of crime": PERSON,
    "tangible property, other than real estate, including contraband": OTHER,
    "chemical company": ORGANIZATION,
    "child, children, including adopted or illegitimate": PERSON,
    "religious organization, institution, or person": OTHER,
    "private club or facility": FACILITY,
    "coal company or coal mine operator": OTHER,
    "computer business or manufacturer, hardware or software": ORGANIZATION,
    "consumer, consumer organization": OTHER,
    "creditor, including institution appearing as such; e.g., a finance company": OTHER,
    "person allegedly criminally insane or mentally incompetent to stand trial": PERSON,
    "defendant": OTHER,
    "debtor": OTHER,
    "real estate developer": PERSON,
    "disabled person or disability benefit claimant": PERSON,
    "distributor": OTHER,
    "person subject to selective service, including conscientious objector": PERSON,
    "drug manufacturer": ORGANIZATION,
    "druggist, pharmacist, pharmacy": OTHER,
    "employee, or job applicant, including beneficiaries of": PERSON,
    "employer-employee trust agreement, employee health and welfare fund, or multi-employer pension plan": OTHER,
    "electric equipment manufacturer": ORGANIZATION,
    "electric or hydroelectric power utility, power cooperative, or gas and electric company": ORGANIZATION,
    "eleemosynary institution or person": OTHER,
    "environmental organization": ORGANIZATION,
    "employer. If employer's relations with employees are governed by the nature of the employer's business (e.g., railroad, boat), rather than labor law generally, the more specific designation is used in place of Employer.":PERSON,
    "farmer, farm worker, or farm organization": PERSON,
    "father": PERSON,
    "female employee or job applicant": PERSON,
    "female": PERSON,
    "movie, play, pictorial representation, theatrical production, actor, or exhibitor or distributor of": OTHER,
    "fisherman or fishing company": OTHER,
    "food, meat packing, or processing company, stockyard": OTHER,
    "foreign (non-American) nongovernmental entity": OTHER,
    "franchiser": PERSON,
    "franchisee": ORGANIZATION,
    "lesbian, gay, bisexual, transexual person or organization": OTHER,
    "person who guarantees another's obligations": PERSON,
    "handicapped individual, or organization of devoted to": OTHER,
    "health organization or person, nursing home, medical clinic or laboratory, chiropractor": OTHER,
    "heir, or beneficiary, or person so claiming to be": OTHER,
    "hospital, medical center": OTHER,
    "husband, or ex-husband": PERSON,
    "involuntarily committed mental patient": PERSON,
    "Indian, including Indian tribe or nation": OTHER,
    "insurance company, or surety": ORGANIZATION,
    "inventor, patent assigner, trademark owner or holder": OTHER,
    "investor": PERSON,
    "injured person or legal entity, nonphysically and non-employment related": OTHER,
    "juvenile": OTHER,
    "government contractor": OTHER,
    "holder of a license or permit, or applicant therefor": OTHER,
    "magazine": ORGANIZATION,
    "male": PERSON,
    "medical or Medicaid claimant": PERSON,
    "medical supply or manufacturing co.": OTHER,
    "racial or ethnic minority employee or job applicant": PERSON,
    "minority female employee or job applicant": PERSON,
    "manufacturer": ORGANIZATION,
    "management, executive officer, or director, of business entity": PERSON,
    "military personnel, or dependent of, including reservist": PERSON,
    "mining company or miner, excluding coal, oil, or pipeline company": ORGANIZATION,
    "mother": PERSON,
    "auto manufacturer": ORGANIZATION,
    "newspaper, newsletter, journal of opinion, news service": ORGANIZATION,
    "radio and television network, except cable tv": ORGANIZATION,
    "nonprofit organization or business": ORGANIZATION,
    "nonresident": PERSON,
    "nuclear power plant or facility": FACILITY,
    "owner, landlord, or claimant to ownership, fee interest, or possession of land as well as chattels": OTHER,
    "shareholders to whom a tender offer is made": OTHER,
    "tender offer": OTHER,
    "oil company, or natural gas producer": ORGANIZATION,
    "elderly person, or organization dedicated to the elderly": OTHER,
    "out of state noncriminal defendant": PERSON,
    "political action committee": ORGANIZATION,
    "parent or parents": PERSON,
    "parking lot or service": OTHER,
    "patient of a health professional": PERSON,
    "telephone, telecommunications, or telegraph company": ORGANIZATION,
    "physician, MD or DO, dentist, or medical society": OTHER,
    "public interest organization": ORGANIZATION,
    "physically injured person, including wrongful death, who is not an employee": PERSON,
    "pipe line company": ORGANIZATION,
    "package, luggage, container": OTHER,
    "political candidate, activist, committee, party, party member, organization, or elected official": OTHER,
    "indigent, needy, welfare recipient": PERSON,
    "indigent defendant": PERSON,
    "private person": PERSON,
    "prisoner, inmate of penal institution": PERSON,
    "professional organization, business, or person": OTHER,
    "probationer, or parolee": OTHER,
    "protester, demonstrator, picketer or pamphleteer (non-employment related), or non-indigent loiterer": PERSON,
    "public utility": OTHER,
    "publisher, publishing company": ORGANIZATION,
    "radio station": ORGANIZATION,
    "racial or ethnic minority": PERSON,
    "person or organization protesting racial or ethnic segregation or discrimination": OTHER,
    "racial or ethnic minority student or applicant for admission to an educational institution": PERSON,
    "realtor": PERSON,
    "journalist, columnist, member of the news media": PERSON,
    "resident": PERSON,
    "restaurant, food vendor": ORGANIZATION,
    "retarded person, or mental incompetent": PERSON,
    "retired or former employee": PERSON,
    "railroad": FACILITY,
    "private school, college, or university": ORGANIZATION,
    "seller or vendor": PERSON,
    "shipper, including importer and exporter": OTHER,
    "shopping center, mall": FACILITY,
    "spouse, or former spouse": PERSON,
    "stockholder, shareholder, or bondholder": OTHER,
    "retail business or outlet": ORGANIZATION,
    "student, or applicant for admission to an educational institution": PERSON,
    "taxpayer or executor of taxpayer's estate, federal only": PERSON,
    "tenant or lessee": OTHER,
    "theater, studio": FACILITY,
    "forest products, lumber, or logging company": ORGANIZATION,
    "person traveling or wishing to travel abroad, or overseas travel agent": PERSON,
    "trucking company, or motor carrier": ORGANIZATION,
    "television station": ORGANIZATION,
    "union member": OTHER,
    "unemployed person or unemployment compensation applicant or claimant": PERSON,
    "union, labor organization, or official of": ORGANIZATION,
    "veteran": PERSON,
    "voter, prospective voter, elector, or a nonelective official seeking reapportionment or redistricting of legislative districts (POL)": PERSON,
    "wholesale trade": OTHER,
    "wife, or ex-wife": PERSON,
    "witness, or person under subpoena": PERSON,
    "network": OTHER,
    "slave": PERSON,
    "slave-owner": PERSON,
    "bank of the united states": ORGANIZATION,
    "timber company": ORGANIZATION,
    "u.s. job applicants or employees": PERSON,
    "Army and Air Force Exchange Service": OTHER,
    "Atomic Energy Commission": PUBLIC_ENTITY,
    "Secretary or administrative unit or personnel of the U.S. Air Force": PUBLIC_ENTITY,
    "Department or Secretary of Agriculture": PUBLIC_ENTITY,
    "Alien Property Custodian": OTHER,
    "Secretary or administrative unit or personnel of the U.S. Army": PUBLIC_ENTITY,
    "Board of Immigration Appeals": PUBLIC_ENTITY,
    "Bureau of Indian Affairs": PUBLIC_ENTITY,
    "Bonneville Power Administration": PUBLIC_ENTITY,
    "Benefits Review Board": PUBLIC_ENTITY,
    "Civil Aeronautics Board": PUBLIC_ENTITY,
    "Bureau of the Census": PUBLIC_ENTITY,
    "Central Intelligence Agency": PUBLIC_ENTITY,
    "Commodity Futures Trading Commission": OTHER,
    "Department or Secretary of Commerce": PUBLIC_ENTITY,
    "Comptroller of Currency": OTHER,
    "Consumer Product Safety Commission": PUBLIC_ENTITY,
    "Civil Rights Commission": PUBLIC_ENTITY,
    "Civil Service Commission, U.S.": PUBLIC_ENTITY,
    "Customs Service or Commissioner of Customs": OTHER,
    "Defense Base Closure and REalignment Commission": OTHER,
    "Drug Enforcement Agency": PUBLIC_ENTITY,
    "Department or Secretary of Defense (and Department or Secretary of War)": PUBLIC_ENTITY,
    "Department or Secretary of Energy": PUBLIC_ENTITY,
    "Department or Secretary of the Interior": PUBLIC_ENTITY,
    "Department of Justice or Attorney General": PUBLIC_ENTITY,
    "Department or Secretary of State": PUBLIC_ENTITY,
    "Department or Secretary of Transportation": PUBLIC_ENTITY,
    "Department or Secretary of Education": PUBLIC_ENTITY,
    "U.S. Employees' Compensation Commission, or Commissioner": PUBLIC_ENTITY,
    "Equal Employment Opportunity Commission": PUBLIC_ENTITY,
    "Environmental Protection Agency or Administrator": PUBLIC_ENTITY,
    "Federal Aviation Agency or Administration": PUBLIC_ENTITY,
    "Federal Bureau of Investigation or Director": PUBLIC_ENTITY,
    "Federal Bureau of Prisons": PUBLIC_ENTITY,
    "Farm Credit Administration": PUBLIC_ENTITY,
    "Federal Communications Commission (including a predecessor, Federal Radio Commission)": OTHER,
    "Federal Credit Union Administration": PUBLIC_ENTITY,
    "Food and Drug Administration": PUBLIC_ENTITY,
    "Federal Deposit Insurance Corporation": PUBLIC_ENTITY,
    "Federal Energy Administration": PUBLIC_ENTITY,
    "Federal Election Commission": PUBLIC_ENTITY,
    "Federal Energy Regulatory Commission": PUBLIC_ENTITY,
    "Federal Housing Administration": PUBLIC_ENTITY,
    "Federal Home Loan Bank Board": PUBLIC_ENTITY,
    "Federal Labor Relations Authority": PUBLIC_ENTITY,
    "Federal Maritime Board": PUBLIC_ENTITY,
    "Federal Maritime Commission": PUBLIC_ENTITY,
    "Farmers Home Administration": PUBLIC_ENTITY,
    "Federal Parole Board": PUBLIC_ENTITY,
    "Federal Power Commission": PUBLIC_ENTITY,
    "Federal Railroad Administration": PUBLIC_ENTITY,
    "Federal Reserve Board of Governors": PUBLIC_ENTITY,
    "Federal Reserve System": PUBLIC_ENTITY,
    "Federal Savings and Loan Insurance Corporation": PUBLIC_ENTITY,
    "Federal Trade Commission": PUBLIC_ENTITY,
    "Federal Works Administration, or Administrator": PUBLIC_ENTITY,
    "General Accounting Office": PUBLIC_ENTITY,
    "Comptroller General": OTHER,
    "General Services Administration": PUBLIC_ENTITY,
    "Department or Secretary of Health, Education and Welfare": PUBLIC_ENTITY,
    "Department or Secretary of Health and Human Services": PUBLIC_ENTITY,
    "Department or Secretary of Housing and Urban Development": PUBLIC_ENTITY,
    "Interstate Commerce Commission": PUBLIC_ENTITY,
    "Indian Claims Commission": PUBLIC_ENTITY,
    "Immigration and Naturalization Service, or Director of, or District Director of, or Immigration and Naturalization Enforcement": PUBLIC_ENTITY,
    "Internal Revenue Service, Collector, Commissioner, or District Director of": PUBLIC_ENTITY,
    "Information Security Oversight Office": ORGANIZATION,
    "Department or Secretary of Labor": PUBLIC_ENTITY,
    "Loyalty Review Board": ORGANIZATION,
    "Legal Services Corporation": ORGANIZATION,
    "Merit Systems Protection Board": ORGANIZATION,
    "Multistate Tax Commission": PUBLIC_ENTITY,
    "National Aeronautics and Space Administration": PUBLIC_ENTITY,
    "Secretary or administrative unit of the U.S. Navy": PUBLIC_ENTITY,
    "National Credit Union Administration": PUBLIC_ENTITY,
    "National Endowment for the Arts": PUBLIC_ENTITY,
    "National Enforcement Commission": PUBLIC_ENTITY,
    "National Highway Traffic Safety Administration": PUBLIC_ENTITY,
    "National Labor Relations Board, or regional office or officer": PUBLIC_ENTITY,
    "National Mediation Board": PUBLIC_ENTITY,
    "National Railroad Adjustment Board": PUBLIC_ENTITY,
    "Nuclear Regulatory Commission": PUBLIC_ENTITY,
    "National Security Agency": PUBLIC_ENTITY,
    "Office of Economic Opportunity": PUBLIC_ENTITY,
    "Office of Management and Budget": PUBLIC_ENTITY,
    "Office of Price Administration, or Price Administrator": PUBLIC_ENTITY,
    "Office of Personnel Management": PUBLIC_ENTITY,
    "Occupational Safety and Health Administration": PUBLIC_ENTITY,
    "Occupational Safety and Health Review Commission": PUBLIC_ENTITY,
    "Office of Workers' Compensation Programs": PUBLIC_ENTITY,
    "Patent Office, or Commissioner of, or Board of Appeals of": PUBLIC_ENTITY,
    "Pay Board (established under the Economic Stabilization Act of 1970)": PUBLIC_ENTITY,
    "Pension Benefit Guaranty Corporation": PUBLIC_ENTITY,
    "U.S. Public Health Service": PUBLIC_ENTITY,
    "Postal Rate Commission": PUBLIC_ENTITY,
    "Provider Reimbursement Review Board": PUBLIC_ENTITY,
    "Renegotiation Board": PUBLIC_ENTITY,
    "Railroad Adjustment Board": PUBLIC_ENTITY,
    "Railroad Retirement Board": PUBLIC_ENTITY,
    "Subversive Activities Control Board": PUBLIC_ENTITY,
    "Small Business Administration": PUBLIC_ENTITY,
    "Securities and Exchange Commission": PUBLIC_ENTITY,
    "Social Security Administration or Commissioner": PUBLIC_ENTITY,
    "Selective Service System": PUBLIC_ENTITY,
    "Department or Secretary of the Treasury": PUBLIC_ENTITY,
    "Tennessee Valley Authority": PUBLIC_ENTITY,
    "United States Forest Service": PUBLIC_ENTITY,
    "United States Parole Commission": PUBLIC_ENTITY,
    "Postal Service and Post Office, or Postmaster General, or Postmaster": PUBLIC_ENTITY,
    "United States Sentencing Commission": PUBLIC_ENTITY,
    "Veterans' Administration": PUBLIC_ENTITY,
    "War Production Board": PUBLIC_ENTITY,
    "Wage Stabilization Board": PUBLIC_ENTITY,
    "General Land Office of Commissioners": PUBLIC_ENTITY,
    "Transportation Security Administration": PUBLIC_ENTITY,
    "Surface Transportation Board": PUBLIC_ENTITY,
    "U.S. Shipping Board Emergency Fleet Corp.": ORGANIZATION,
    "Reconstruction Finance Corp.": ORGANIZATION,
    "Department or Secretary of Homeland Security": PUBLIC_ENTITY,
    "Unidentifiable": OTHER,
    "International Entity": ORGANIZATION,
}

ISSUE_AREA_MAPPING = {k.lower().replace(" ", "_"): i for i, k in enumerate(ISSUE_AREAS)}

METADATA_FIELDS = ['decisionDirection', 'respondent']

@dataclass
class SCOTUSConfig(datasets.BuilderConfig):
    version:str=None
    name:str = None
    extended_name: str = None


class ScotusDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SCOTUSConfig(
            name='scotus',
            extended_name = 'Supreme Court of United States Dataset',
            version=datasets.Version(_VERSION, ""),
        )
    ]

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "decisionDirection": datasets.Value("string"),
                    "respondent":datasets.Value("string"),
                    'label_ids': datasets.Value('int64'),
                    "labels": datasets.Value("string")
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        downloaded_file = dl_manager.download_and_extract(_DOWNLOAD_URL)
        dataset_root_folder = os.path.join(downloaded_file)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset_root": dataset_root_folder,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset_root": dataset_root_folder,
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset_root": dataset_root_folder,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, dataset_root, split, **kwargs):
        path = os.path.join(dataset_root, 'scotus', f'scotus.{split}.jsonl')
        with jsonlines.open(path) as lines:
            for line in lines:
                attributes = line['attributes']
                iid = attributes['docketId']
                text = line['text']
                label = str(line['label'])
                label_id = ISSUE_AREA_MAPPING[label]
                respondent = RESPONDENT_MAPPING[attributes['respondent']]
                yield iid, {'text': text, 'id': iid, 'labels':label, 'label_ids': label_id,  'decisionDirection': attributes['decisionDirection'], 'respondent': respondent}

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset('/home/npf290/dev/fairlex-wilds/huggingface_scotus/scotus.py')
    dataset['train']
    print()