import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.XMLConstants;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.IOException;

class VoteRow {
    private String Id;
    private String PostId;
    private String VoteTypeId;
    private String CreationDate;

    public VoteRow(String Id, String PostId, String VoteTypeId, String CreationDate) {
        this.Id = Id;
        this.PostId = PostId;
        this.VoteTypeId = VoteTypeId;
        this.CreationDate = CreationDate;
    }

    public String getId() {
        return Id;
    }

    public String getPostId() {
        return PostId;
    }

    public String getVoteTypeId() {
        return VoteTypeId;
    }

    public String getCreationDate() {
        return CreationDate;
    }

    @Override
    public String toString() {
        return "VoteRow{" +
                "Id='" + Id + '\'' +
                ", PostId='" + PostId + '\'' +
                ", VoteTypeId='" + VoteTypeId + '\'' +
                ", CreationDate='" + CreationDate + '\'' +
                '}';
    }
}

public class VotesHandler extends DefaultHandler {
    private static final String ROW = "row";
    private static final String VOTES = "votes";

    @Override
    public void startElement(String uri, String lName, String qName, Attributes attr) throws SAXException {
        if (qName.equals(ROW)) {
            VoteRow voteRow = new VoteRow(
                    attr.getValue("Id"),
                    attr.getValue("PostId"),
                    attr.getValue("VoteTypeId"),
                    attr.getValue("CreationDate")
            );
            System.out.println(voteRow);

        } else if (!qName.equals(VOTES)) {
            throw new SAXException(String.format("Unknown tag %s", qName));
        }

    }

    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
        final long start = System.currentTimeMillis();
        System.out.println(start);
        SAXParserFactory factory = SAXParserFactory.newInstance();
        factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING,false);
        SAXParser saxParser = factory.newSAXParser();
        VotesHandler handler = new VotesHandler();
        saxParser.parse(
                "..\\data\\raw\\stackoverflow.com-Votes\\votes.xml", handler);


        // time end
        final long durationInMilliseconds = System.currentTimeMillis()-start;
        System.out.println("SAX parse took " + durationInMilliseconds + "ms.");
    }
}
