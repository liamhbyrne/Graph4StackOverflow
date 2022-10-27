import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.XMLConstants;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.IOException;

class UserRow {
    private String Id;
    private String Reputation;
    private String CreationDate;
    private String DisplayName;
    private String LastAccessDate;
    private String AboutMe;
    private String Views;
    private String UpVotes;
    private String DownVotes;


    public UserRow(String Id, String Reputation, String CreationDate, String DisplayName,
                   String LastAccessDate, String AboutMe, String Views, String UpVotes,
                   String DownVotes) {
        this.Id = Id;
        this.Reputation = Reputation;
        this.CreationDate = CreationDate;
        this.DisplayName = DisplayName;
        this.LastAccessDate = LastAccessDate;
        this.AboutMe = AboutMe;
        this.Views = Views;
        this.UpVotes = UpVotes;
        this.DownVotes = DownVotes;

    }

    public String getId() {
        return Id;
    }

    public String getReputation() {
        return Reputation;
    }

    public String getCreationDate() {
        return CreationDate;
    }

    public String getDisplayName() {
        return DisplayName;
    }

    public String getLastAccessDate() {
        return LastAccessDate;
    }

    public String getAboutMe() {
        return AboutMe;
    }

    public String getViews() {
        return Views;
    }

    public String getUpVotes() {
        return UpVotes;
    }

    public String getDownVotes() {
        return DownVotes;
    }

    @Override
    public String toString() {
        return "UserRow{" +
                "Id='" + Id + '\'' +
                ", Reputation='" + Reputation + '\'' +
                ", CreationDate='" + CreationDate + '\'' +
                ", DisplayName='" + DisplayName + '\'' +
                ", LastAccessDate='" + LastAccessDate + '\'' +
                ", AboutMe='" + AboutMe + '\'' +
                ", Views='" + Views + '\'' +
                ", UpVotes='" + UpVotes + '\'' +
                ", DownVotes='" + DownVotes + '\'' +
                '}';
    }
}

public class UsersHandler extends DefaultHandler {
    private static final String ROW = "row";
    private static final String USERS = "users";

    @Override
    public void startElement(String uri, String lName, String qName, Attributes attr) throws SAXException {
        if (qName.equals(ROW)) {
            UserRow userRow = new UserRow(
                    attr.getValue("Id"),
                    attr.getValue("Reputation"),
                    attr.getValue("CreationDate"),
                    attr.getValue("DisplayName"),
                    attr.getValue("LastAccessDate"),
                    attr.getValue("AboutMe"),
                    attr.getValue("Views"),
                    attr.getValue("UpVotes"),
                    attr.getValue("DownVotes")
            );
            System.out.println(userRow);

        } else if (!qName.equals(USERS)) {
            throw new SAXException(String.format("Unknown tag %s", qName));
        }

    }

    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
        final long start = System.currentTimeMillis();
        System.out.println(start);
        SAXParserFactory factory = SAXParserFactory.newInstance();
        factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING,false);
        SAXParser saxParser = factory.newSAXParser();
        UsersHandler handler = new UsersHandler();
        saxParser.parse(
                "..\\data\\raw\\stackoverflow.com-Users\\users.xml", handler);


        // time end
        final long durationInMilliseconds = System.currentTimeMillis()-start;
        System.out.println("SAX parse took " + durationInMilliseconds + "ms.");
    }
}
