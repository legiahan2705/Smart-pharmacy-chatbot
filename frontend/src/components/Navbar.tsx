import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Home, MessageSquare } from "lucide-react";

const Navbar = () => {
  return (
    <header className="bg-[#DEECFF] py-6 flex justify-center">
      <nav className="bg-background rounded-xl shadow-lg flex items-center justify-between px-6 py-3 w-full max-w-xl text-[#001A61]">
        {/* Left - Home */}
        <Link 
          href="/" 
          className="flex items-center gap-2 text-[#001A61] font-semibold hover:text-black transition-all duration-300"
        >
          <Home className="w-5 h-5" />
          <span >Home</span>
        </Link>

        {/* Right - Chatbot & Sign in */}
        <div className="flex items-center gap-4">
          <Link 
            href="/chatbot" 
            className="flex items-center gap-2 text-[#001A61] font-semibold hover:text-black transition-all duration-300"
          >
            <MessageSquare className="w-4 h-4" />
            <span>Chatbot</span>
          </Link>
          <Link href="/signin">
            <Button size="sm" className="text-[#001A61] font-semibold hover:bg-blue-50 hover:text-[#001A61] transition-all duration-300 bg-[#DEECFF]">
              Sign in
            </Button>
          </Link>
        </div>
      </nav>
    </header>
  );
};

export default Navbar;