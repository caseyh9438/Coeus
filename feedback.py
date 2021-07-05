import rdkit
from rdkit import Chem
from rdkit import rdBase
from rdkit.six import iteritems
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumHBD, CalcNumHBA
from rdkit.Chem.Descriptors import ExactMolWt, MolLogP, MolMR
from data_module import SmilesDataModule


class MoleculeValidationAndFeedback(SmilesDataModule):
 
  def __init__(self, sa_score = None, molecules = None, tokenizer = None, max_length = None):
    self.molecules = molecules
    self.max_length = max_length
    self.tokenizer = tokenizer
    self.sa_score = sa_score
    self.probability_dict = None
    self.anti_proteins = {'MLH1': 'MSFVAGVIRRLDETVVNRIAAGEVIQRPANAIKEMIENCLDAKSTSIQVIVKEGGLKLIQIQDNGTGIRKEDLDIVCERFTTSKLQSFEDLASISTYGFRGEALASISHVAHVTITTKTADGKCAYRASYSDGKLKAPPKPCAGNQGTQITVEDLFYNIATRRKALKNPSEEYGKILEVVGRYSVHNAGISFSVKKQGETVADVRTLPNASTVDNIRSIFGNAVSRELIEIGCEDKTLAFKMNGYISNANYSVKKCIFLLFINHRLVESTSLRKAIETVYAAYLPKNTHPFLYLSLEISPQNVDVNVHPTKHEVHFLHEESILERVQQHIESKLLGSNSSRMYFTQTLLPGLAGPSGEMVKSTTSLTSSSTSGSSDKVYAHQMVRTDSREQKLDAFLQPLSKPLSSQPQAIVTEDKTDISSGRARQQDEEMLELPAPAEVAAKNQSLEGDTTKGTSEMSEKRGPTSSNPRKRHREDSDVEMVEDDSRKEMTAACTPRRRIINLTSVLSLQEEINEQGHEVLREMLHNHSFVGCVNPQWALAQHQTKLYLLNTTKLSEELFYQILIYDFANFGVLRLSEPAPLFDLAMLALDSPESGWTEEDGPKEGLAEYIVEFLKKKAEMLADYFSLEIDEEGNLIGLPLLIDNYVPPLEGLPIFILRLATEVNWDEEKECFESLSKECAMFYSIRKQYISEESTLSGQQSEVPGSIPNSWKWTVEHIVYKALRSHILPPKHFTEDGNILQLANLPDLYKVFERC'}
    self.character_list = self.set_character_list(self.molecules)
    self.binding_model = models.model_pretrained(model = 'MPNN_CNN_BindingDB')
    self.pocket = "GGNLLVLVDQHAAHERIRLEQLII"
    self.amino_acid_target = "MIKCLSVEVQAKLRSGLAISSLGQCVEELALNSIDAEAKCVAVRVNMETFQVQVIDNGFGMGSDDVEKVGNRYFTSKCHSVQDLENPRFYGFRGEALANIADMASAVEISSKKNRTMKTFVKLFQSGKALKACEADVTRASAGTTVTVYNLFYQLPVRRKCMDPRLEFEKVRQRIEALSLMHPSISFSLRNDVSGSMVLQLPKTKDVCSRFCQIYGLGKSQKLREISFKYKEFELSGYISSEAHYNKNMQFLFVNKRLVLRTKLHKLIDFLLRKESIICKPKNGPTSRQMNSSLRHRSTPELYGIYVINVQCQFCEYDVCMEPAKTLIEFQNWDTLLFCIQEGVKMFLKQEKLFVELSGEDIKEFSEDNGFSLFDATLQKRVTSDERSNFQEACNNILDSYEMFNLQSKAVKRKTTAENVNTQSSRDSEATRKNTNDAFLYIYESGGPGHSKMTEPSLQNKDSSCSESKMLEQETIVASEAGENEKHKKSFLEHSSLENPCGTSLEMFLSPFQTPCHFEESGQDLEIWKESTTVNGMAANILKNNRIQNQPKRFKDATEVGCQPLPFATTLWGVHSAQTEKEKKKESSNCGRRNVFSYGRVKLCSTGFITHVVQNEKTKSTETEHSFKNYVRPGPTRAQETFGNRTRHSVETPDIKDLASTLSKESGQLPNKKNCRTNISYGLENEPTATYTMFSAFQEGSKKSQTDCILSDTSPSFPWYRHVSNDSRKTDKLIGFSKPIVRKKLSLSSQLGSLEKFKRQYGKVENPLDTEVEESNGVTTNLSLQVEPDILLKDKNRLENSDVCKITTMEHSDSDSSCQPASHILNSEKFPFSKDEDCLEQQMPSLRESPMTLKELSLFNRKPLDLEKSSESLASKLSRLKGSERETQTMGMMSRFNELPNSDSSRKDSKLCSVLTQDFCMLFNNKHEKTENGVIPTSDSATQDNSFNKNSKTHSNSNTTENCVISETPLVLPYNNSKVTGKDSDVLIRASEQQIGSLDSPSGMLMNPVEDATGDQNGICFQSEESKARACSETEESNTCCSDWQRHFDVALGRMVYVNKMTGLSTFIAPTEDIQAACTKDLTTVAVDVVLENGSQYRCQPFRSDLVLPFLPRARAERTVMRQDNRDTVDDTVSSESLQSLFSEWDNPVFARYPEVAVDVSSGQAESLAVKIHNILYPYRFTKGMIHSMQVLQQVDNKFIACLMSTKTEENGEAGGNLLVLVDQHAAHERIRLEQLIIDSYEKQQAQGSGRKKLLSSTLIPPLEITVTEEQRRLLWCYHKNLEDLGLEFVFPDTSDSLVLVGKVPLCFVEREANELRRGRSTVTKSIVEEFIREQLELLQTTGGIQGTLPLTVQKVLASQACHGAIKFNDGLSLQESCRLIEALSSCQLPFQCAHGRPSMLPLADIDHLEQEKQIKPNLTKLRKMAQAWRLFGKAECDTRQSLQQSMPPCEPP"
    
    """
    Below is the MLH1 & MLH3 protein complex in fasta format

    Our Target -> a specific binding pocket on MLH3 to inhibit activity with hope of preventing somatic expansion.
    I've separated the sequence for the endonuclease

    >sp|Q9UHC1|MLH3_HUMAN DNA mismatch repair protein Mlh3 OS=Homo sapiens OX=9606 GN=MLH3 PE=1 SV=3
    MIKCLSVEVQAKLRSGLAISSLGQCVEELALNSIDAEAKCVAVRVNMETFQVQVIDNGFG
    MGSDDVEKVGNRYFTSKCHSVQDLENPRFYGFRGEALANIADMASAVEISSKKNRTMKTF
    VKLFQSGKALKACEADVTRASAGTTVTVYNLFYQLPVRRKCMDPRLEFEKVRQRIEALSL
    MHPSISFSLRNDVSGSMVLQLPKTKDVCSRFCQIYGLGKSQKLREISFKYKEFELSGYIS
    SEAHYNKNMQFLFVNKRLVLRTKLHKLIDFLLRKESIICKPKNGPTSRQMNSSLRHRSTP
    ELYGIYVINVQCQFCEYDVCMEPAKTLIEFQNWDTLLFCIQEGVKMFLKQEKLFVELSGE
    DIKEFSEDNGFSLFDATLQKRVTSDERSNFQEACNNILDSYEMFNLQSKAVKRKTTAENV
    NTQSSRDSEATRKNTNDAFLYIYESGGPGHSKMTEPSLQNKDSSCSESKMLEQETIVASE
    AGENEKHKKSFLEHSSLENPCGTSLEMFLSPFQTPCHFEESGQDLEIWKESTTVNGMAAN
    ILKNNRIQNQPKRFKDATEVGCQPLPFATTLWGVHSAQTEKEKKKESSNCGRRNVFSYGR
    VKLCSTGFITHVVQNEKTKSTETEHSFKNYVRPGPTRAQETFGNRTRHSVETPDIKDLAS
    TLSKESGQLPNKKNCRTNISYGLENEPTATYTMFSAFQEGSKKSQTDCILSDTSPSFPWY
    RHVSNDSRKTDKLIGFSKPIVRKKLSLSSQLGSLEKFKRQYGKVENPLDTEVEESNGVTT
    NLSLQVEPDILLKDKNRLENSDVCKITTMEHSDSDSSCQPASHILNSEKFPFSKDEDCLE
    QQMPSLRESPMTLKELSLFNRKPLDLEKSSESLASKLSRLKGSERETQTMGMMSRFNELP
    NSDSSRKDSKLCSVLTQDFCMLFNNKHEKTENGVIPTSDSATQDNSFNKNSKTHSNSNTT
    ENCVISETPLVLPYNNSKVTGKDSDVLIRASEQQIGSLDSPSGMLMNPVEDATGDQNGIC
    FQSEESKARACSETEESNTCCSDWQRHFDVALGRMVYVNKMTGLSTFIAPTEDIQAACTK
    DLTTVAVDVVLENGSQYRCQPFRSDLVLPFLPRARAERTVMRQDNRDTVDDTVSSESLQS
    LFSEWDNPVFARYPEVAVDVSSGQAESLAVKIHNILYPYRFTKGMIHSMQVLQQVDNKFI
    ACLMSTKTEENGEA-------GGNLLVLVDQHAAHERIRLEQLII----------DSYEK
    QQAQGSGRKKLLSSTLIPPLEITVTEEQRRLLWCYHKNLEDLGLEFVFPDTSDSLVLVGK
    VPLCFVEREANELRRGRSTVTKSIVEEFIREQLELLQTTGGIQGTLPLTVQKVLASQACH
    GAIKFNDGLSLQESCRLIEALSSCQLPFQCAHGRPSMLPLADIDHLEQEKQIKPNLTKLR
    KMAQAWRLFGKAECDTRQSLQQSMPPCEPP
    """

  def get_starting_character(self, char_list):

    """This function determines probabilities for each starting 
        letter and picks one based on the likley-hood a molecule would 
        start with that letter. This is then feed into the model 
        as a starting character for molecule generation."""

    self.probability_dict = pd.read_csv('https://storage.googleapis.com/htr1/LettersProbabilities1M.csv').drop(columns=['Unnamed: 0'])
    # self.probability_dict = {character: char_list.count(character) / (len(char_list) + 1) for character in char_list} if self.probability_dict is None else self.probability_dict
    letters = list(self.probability_dict.keys())
    probabilities = np.array(list(self.probability_dict.values()))
    probabilities /= probabilities.sum()
    
    # SAVE LETTERS AND PROBABILITIES TO CSV TO SAVE TIME ON LATER RUNS
    pd.DataFrame(data=list(zip(letters, probabilities)), columns=['letters', 'probabilities']).to_csv('LettersProbabilities1M.csv')
    return np.random.choice(letters, p = probabilities.tolist())

  def run_utility_check(self, molecule):

    if self.is_molecule_valid(molecule) is False:
      return {False: "Failed is_molecule_valid"}
    elif self.is_molecule_novel(molecule) is False:
      return {False: "Failed is_molecule_novel"}
    elif self.is_molecule_synthesizable(molecule) is False:
      return {False: "Failed is_molecule_synthesizable"}
    elif self.is_molecule_druglike(molecule) is False:
      return {False: "Failed is_molecule_druglike"}
    elif self.can_molecule_pass_bbb(molecule) is False:
      return {False: "Failed can_molecule_pass_bbb"}
    else:
      return {True: self.get_binding_prediction(molecule = molecule)}

  def run_feedback_pipeline(self, model = None, molecules_to_generate = None):
    """This returns a subset that pass above filters and provides binding scores"""
    list_new_molecules = self.generate_molecules(model = model, molecules_to_generate = molecules_to_generate)
    score, drug_meets_desires = (lambda molecule: list(self.run_utility_check(molecule).values())[0], lambda molecule: list(self.run_utility_check(molecule).keys())[0])
    list_new_molecules = [smile for smile in list_new_molecules if len(smile) < self.max_length]
    return [(score(molecule), molecule) for molecule in list_new_molecules if drug_meets_desires(molecule) is True and isinstance(score(molecule), int)]

  def get_anti_protein_binding(self, molecule): 
    """This functions purpose to ensure that the molecule has a low 
       binding score on select proteins such as MLH1 and PMS2"""
    return [self.get_binding_prediction(molecule = molecule, 
                                        protein = anti_prot) for anti_prot in list(self.anti_proteins.values())]

  def get_binding_prediction(self, molecule = None, protein = None):
    return self.binding_model.predict(utils.data_process(X_drug = [molecule], X_target = [self.pocket if protein is None else protein], y = [0],
                                            drug_encoding = 'MPNN', target_encoding = 'CNN', 
                                            split_method = 'no_split'))[0]

  def get_binding_separation(self):
    #  BINDING PREDICTIONS - ANTI BINDING PREDICTIONS
    pass

  def is_molecule_valid(self, molecule):
    return Chem.MolFromSmiles(molecule) is not None

  def is_molecule_novel(self, molecule):
    return self.molecules.count(molecule) == 0

  def is_molecule_nontoxic(self, molecule):
    pass
  
  def is_molecule_synthesizable(self, molecule):
    return self.sa_score(molecule)
  
  def is_molecule_druglike(self, molecule):

    """
    We're following Lipinski's Rule of 5 along with the Ghose filter 
    to determine druglikeness.

    Number of Hydrogen Bond Donnors (HBD) -->              < 5
    Number of Hydrogen Bond Acceptor (HBA) -->             < 10
    Logarithm of Partition Coefficient (LogP) -->          -0.4 - 5.6
    Molecule Weight (MW) -->                               160 - 480 g/mol
    Molar Regractivity (MR) -->                            40 - 130
    Number of Atom (ATOMS) -->                             20 - 70
    """
    molecule = Chem.MolFromSmiles(molecule)
    MW, LogP, HBD, HBA, ATOMS, MR = (ExactMolWt(molecule), MolLogP(molecule), 
                                     CalcNumHBD(molecule), CalcNumHBA(molecule), 
                                     molecule.GetNumAtoms(), MolMR(molecule))
    
    meets_mw = MW > 160 and MW < 480
    meets_logp = LogP > -0.4 and LogP < 5.6
    meets_hbd = HBD < 5
    meets_hba = HBA < 10
    meets_mr = MR > 40 and MR < 130
    meets_num_atoms = ATOMS > 20 and ATOMS < 70

    return meets_mw and meets_logp and meets_hbd and meets_hba and meets_mr and meets_num_atoms

  def can_molecule_pass_bbb(self, molecule):
    """ bbb --> Blood Brain Barrier """
    molecule = Chem.MolFromSmiles(molecule)
    return CalcTPSA(molecule) < 90

  def set_character_list(self, molecules):
    return list(molecule[0] for molecule in molecules)

  def generate_molecules(self, model = None, molecules_to_generate = None):
    results = []
    model.eval()
    for num in range(molecules_to_generate):

      starting_char = self.tokenizer.encode(self.get_starting_character(char_list = self.character_list))

      sample_outputs = model.generate( 
              bos_token_id=torch.tensor([starting_char]).to(torch.device('cuda')),
              do_sample=True,   
              top_k=50, 
              max_length = 250,
              top_p=0.95, 
              num_return_sequences=1)

      for i, sample_output in enumerate(sample_outputs):
          result = self.tokenizer.decode(sample_output, skip_special_tokens = True)
          result = ' '.join(result)
          result_ = result.replace(' ', '')
          print('\nResults: {}\n'.format(result_))
          results.append(result)
    model.train()
    return results
